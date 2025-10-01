class AAttn(nn.Module):
    """
    Five-region area-attention module.

    This module splits the input feature map into five distinct regions
    (four corners and one central region), applies self-attention within
    each region independently, and then merges the attended outputs back
    into the original spatial layout.

    Key idea:
        - Localize attention to spatial regions (instead of full map)
          for computational efficiency and better locality modeling.
        - Add lightweight position encoding to preserve spatial structure.
    """
    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """
        Args:
            dim (int): Input channel dimension.
            num_heads (int): Number of attention heads.
            area (int): Compatibility placeholder for future extension
                        (currently not used).
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.area = area

        # Linear projections for q, k, v (implemented as 1x1 conv).
        self.qkv = Conv(dim, dim * 3, 1, act=False)
        # Final projection after attention fusion.
        self.proj = Conv(dim, dim, 1, act=False)
        # Depthwise convolutional positional encoding.
        self.pe = Conv(dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of area-attention.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            torch.Tensor: Feature map of the same shape after area attention.
        """
        B, C, H, W = x.shape
        h_half, w_half = H // 2, W // 2

        # Partition into 5 regions (4 corners + center).
        regions = [
            x[:, :, :h_half, :w_half],                   # top-left
            x[:, :, :h_half, w_half:],                   # top-right
            x[:, :, h_half:, :w_half],                   # bottom-left
            x[:, :, h_half:, w_half:],                   # bottom-right
            x[:, :, h_half//2:h_half//2+h_half,
                 w_half//2:w_half//2+w_half]             # center
        ]

        outputs = []
        for r in regions:
            B_r, C_r, H_r, W_r = r.shape
            N = H_r * W_r

            # QKV projection and reshape
            qkv = self.qkv(r).flatten(2).transpose(1, 2)  # (B_r, N, 3*C)
            q, k, v = (
                qkv.view(B_r, N, self.num_heads, self.head_dim * 3)
                .permute(0, 2, 3, 1)
                .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
            )

            # Scaled dot-product attention
            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            attn = attn.softmax(dim=-1)
            out = v @ attn.transpose(-2, -1)  # (B_r, num_heads, head_dim, N)

            # Reshape back to (B, C, H, W)
            out = out.permute(0, 3, 1, 2).contiguous()
            out = out.view(B_r, C_r, H_r, W_r)

            # Add position encoding
            v_reshape = v.permute(0, 3, 1, 2).contiguous().view(B_r, C_r, H_r, W_r)
            out = out + self.pe(v_reshape)

            outputs.append(out)

        # Merge the five attended regions back
        out_full = x.clone()
        out_full[:, :, :h_half, :w_half] = outputs[0]
        out_full[:, :, :h_half, w_half:] = outputs[1]
        out_full[:, :, h_half:, :w_half] = outputs[2]
        out_full[:, :, h_half:, w_half:] = outputs[3]

        # Blend center region with weighted averaging
        ch, cw = h_half // 2, w_half // 2
        out_full[:, :, ch:ch + h_half, cw:cw + w_half] = (
            out_full[:, :, ch:ch + h_half, cw:cw + w_half] * 0.8 + outputs[4] * 0.2
        )

        return self.proj(out_full)


class ABlock(nn.Module):
    """
    Area-attention transformer block.

    This block applies area-based attention followed by a lightweight
    feed-forward network (implemented as two 1x1 convolutions).
    Residual connections are used in both sublayers.

    Example:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> out = block(x)
        >>> out.shape
        torch.Size([1, 256, 32, 32])
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        super().__init__()
        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            Conv(dim, mlp_hidden_dim, 1),
            Conv(mlp_hidden_dim, dim, 1, act=False)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.mlp(x)


class CAM(nn.Module):
    """
    C2f-style Area-Attention Module.

    This module integrates area-attention blocks into the CSP-style C2f
    architecture for enhanced feature representation. It can operate in
    two modes:
        - Area-attention mode (with ABlock)
        - Standard convolution mode (with C3k blocks)

    Optionally includes a residual scaling parameter gamma for adaptive
    residual learning.
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            a2 (bool): If True, use area-attention blocks. If False, use C3k.
            area (int): Number of areas for attention division.
            residual (bool): Enable learnable residual scaling.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Expansion factor for hidden channels.
            g (int): Group count for convolutions.
            shortcut (bool): Enable shortcut in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 32 == 0, "Hidden dimension must be divisible by 32 for multi-head attention."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(1, c2, 1, 1), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma * y
        return y

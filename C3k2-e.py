from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # final output conv
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer using chunk()."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottlenecke(nn.Module):
    """Bottleneck-e: Consists of a 3x3 depthwise conv → 1x1 conv → 3x3 conv."""

    def __init__(self, c1, c2, shortcut=True, g=1):
        super().__init__()
        self.add = shortcut and c1 == c2

        # First stage: depthwise separable convolution (3x3 DW + 1x1 PW)
        self.dwconv = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False),
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )

        # Second stage: standard 3x3 convolution
        self.cv2 = nn.Conv2d(c2, c2, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.dwconv(x)
        y = self.act2(self.bn2(self.cv2(y)))
        if self.add:
            y = y + x  # residual connection
        return y


class C3ke(nn.Module):
    """CSP Bottleneck with 3 convolutions, using Bottleneck-e internally."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)

        # Split path convolutions
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        # Bottleneck-e sequence
        self.m = nn.Sequential(*(Bottlenecke(c_, c_, shortcut, g) for _ in range(n)))

    def forward(self, x):
        y1 = self.m(self.cv1(x))  # processed branch
        y2 = self.cv2(x)          # shortcut branch
        return self.cv3(torch.cat((y1, y2), 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions, using either C3k or Bottleneck."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks instead of Bottleneck.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k2e(C3k2):
    """C3k2-e: Faster CSP Bottleneck variant using Bottleneck-e instead of Bottleneck."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)

        self.m = nn.ModuleList(
            C3ke(self.c, self.c, 2, shortcut, g) if c3k else Bottlenecke(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

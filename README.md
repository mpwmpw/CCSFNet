## 1.This is the overall architecture of CCSFNet.

![Logo](p1.png)

## 2. To achieve a deep learning model for safflower maturity detection that balances accuracy and real-time performance, we designed a new attention mechanism (CAM) that additionally focuses on the center region and a lightweight feature extraction module (C3k2-e). CAM not only improves safflower detection, but more specifically, it focuses on improving performance by focusing on the center region. C3k2-e leverages depthwise separable convolutions to achieve model lightweightness and improve performance on edge devices.

<img src="p2.png" width="45%" style="display:inline-block;"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="p3.png" width="45%" style="display:inline-block;">

## 3. Still trying to figure out your loss function? Or struggling with the time it takes to optimize hyperparameters for your deep learning model? FastGA might be able to help! FastGA eliminates the mutation phase in genetic algorithms, significantly speeding up overall parameter tuning without sacrificing performance. However, this may only be suitable for object detection tasks with a small number of categories and minimal scale variation.

<img src="p4.png" width="45%" style="display:inline-block;"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="p5.png" width="45%" style="display:inline-block;">

## 4. Below are our detection results on the large red flower dataset.

<img src="p6.png" width="45%" style="display:inline-block;">

## 5. Due to project process reasons, we cannot make the entire safflower dataset public now, but we have provided a small benchmark that you can download to perform performance testing of the proposed method.

![Logo](p8.png)

## 6. Thanks to the open source of [ultralytics](https://docs.ultralytics.com/zh), many tasks have become simpler and more traceable.

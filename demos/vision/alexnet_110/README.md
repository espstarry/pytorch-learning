# AlexNet

AlexNet 把 CNN 从 LeNet 的小型网络带到了大规模图像分类：网络更深、channel 更多，并使用 ReLU 与 MaxPool。

## 和 LeNet 的主要差别

```text
LeNet:   灰度 32×32, Tanh, AvgPool, 16 个最终 feature maps
AlexNet: RGB 224×224, ReLU, MaxPool, 256 个最终 feature maps, Dropout
```

## 关键 shape

```text
[batch, 3, 224, 224]
-> Conv 3→64, 11×11, stride 4      [batch, 64, 55, 55]
-> MaxPool                         [batch, 64, 27, 27]
-> Conv 64→192                     [batch, 192, 27, 27]
-> MaxPool                         [batch, 192, 13, 13]
-> Conv 192→384 →384 →256          [batch, 256, 13, 13]
-> MaxPool                         [batch, 256, 6, 6]
-> Flatten                         [batch, 9216]
-> Linear 9216→4096→4096→10        [batch, 10]
```

## 运行

```powershell
python demos/vision/alexnet_110/train.py
python demos/vision/alexnet_110/export_tensors.py
```

`export_tensors.py` 生成 `tensor_data.json`，可拖入 [Tensor Viewer](../../tensor_viewer/viewer.html)。这个 demo 使用随机 RGB 图像，只用于理解模型的结构和数据流。

## 用真实 CIFAR-10 测试

```powershell
python demos/vision/alexnet_110/train_cifar10.py
```

脚本会自动下载 CIFAR-10，默认取 1,000 张训练图片和 200 张测试图片，缩放到 `224×224` 后训练 1 个 epoch。CPU 上先用这个小设置验证流程；需要完整训练时可以改参数：

```powershell
python demos/vision/alexnet_110/train_cifar10.py --epochs 10 --train-samples 50000 --test-samples 10000
```

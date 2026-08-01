# LeNet-5

LeNet 是经典 CNN 的入门模型，最适合观察“图像如何逐渐变成分类向量”。

## 结构

```text
输入 [batch, 1, 32, 32]
  -> Conv2d(1, 6, 5)       [batch, 6, 28, 28]
  -> AvgPool2d(2)          [batch, 6, 14, 14]
  -> Conv2d(6, 16, 5)      [batch, 16, 10, 10]
  -> AvgPool2d(2)          [batch, 16, 5, 5]
  -> Flatten               [batch, 400]
  -> Linear(400, 120)
  -> Linear(120, 84)
  -> Linear(84, 10)        [batch, 10]
```

经典 LeNet 使用 `Tanh` 和平均池化；这里保留这个特点，便于理解原始结构。

## 运行

```powershell
python demos/vision/lenet_100/train.py
python demos/vision/lenet_100/export_tensors.py
```

第二条命令会生成 `tensor_data.json`。然后打开 [Tensor Viewer](../../tensor_viewer/viewer.html)，拖入这个 JSON 文件即可查看每一步的 shape 和 tensor 内容。

## 用真实 MNIST 测试

```powershell
python demos/vision/lenet_100/train_mnist.py
```

脚本会自动下载 MNIST，把原始 `28×28` 图片补零到 LeNet 所需的 `32×32`，并在每个 epoch 后打印测试集准确率。

训练后同时导出真实图片的中间 tensor：

```powershell
python demos/vision/lenet_100/train_mnist.py --epochs 1 --export-tensors
```

运行完成后，把 `tensor_data.json` 拖入 [Tensor Viewer](../../tensor_viewer/viewer.html)。如果只想查看真实 MNIST 输入经过未训练 LeNet 的形状和数值，也可以直接运行 `export_tensors.py`。

## 验证底部笔画的作用

```powershell
python demos/vision/lenet_100/probe_bottom_stroke.py
```

脚本会训练一轮，找一张真实数字 `2`，只擦掉最底部 5 行的墨迹，然后比较遮挡前后类别 `2`、`7`、`9` 的 logits。结果保存在 `occlusion_data.json`，也可拖入 Tensor Viewer。

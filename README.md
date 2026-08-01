# PyTorch Model Reading

这个项目只做一件事：用最短的 PyTorch 代码读懂经典模型，并且马上跑起来。

每个模型都是一个独立的小例子，通常只有：

```text
demos/
  vision/lenet_100/
    model.py       # 模型结构
    train.py       # 最小训练和前向示例
    README.md      # 结构与张量尺寸
```

代码刻意不包含日志系统、metric 封装、复杂 DataLoader、K 折验证、实验对比、checkpoint 或配置系统。数据优先使用代码生成的小数据，这样注意力可以放在模型如何构造。

## 运行

在安装 PyTorch 的环境中，从项目根目录运行：

```powershell
python demos/template_000/train.py
```

后续模型会按主题放在这些目录：

```text
vision/       LeNet, AlexNet, VGG, GoogLeNet, ResNet, ViT
detection/    R-CNN, YOLO, SSD, U-Net
sequence/     RNN, LSTM, GRU, Seq2Seq, Attention
nlp/          Transformer, BERT, GPT, T5
generative/   VAE, GAN, Diffusion
graph/        GCN, GAT
multimodal/   CLIP
```

学习顺序：先看 `model.py` 的层和 `forward`，再看 `train.py` 的四步循环：`预测 -> loss -> backward -> optimizer.step()`。

每个 demo 默认还要提供 `export_tensors.py`，把 forward 的关键中间结果导出为 `tensor_data.json`。然后用 [Metric Viewer](viewers/metric_viewer.html) 切换查看 loss、accuracy 和 tensor。具体约定见 [demos/DEMO_RULES.md](demos/DEMO_RULES.md)。

## Colab GPU

用于 VS Code + Colab 的 AlexNet 学习 Notebook 在 [notebooks/alexnet_colab.ipynb](notebooks/alexnet_colab.ipynb)。它会使用 Git 仓库拉取本项目，然后按需运行现有的 `.py` 脚本。

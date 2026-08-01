# Tiny CNN

最小的图像分类例子。输入是 `[batch, 1, 16, 16]`，输出是 `[batch, 2]`。

代码展示了所有训练必需步骤：构造模型、前向、计算交叉熵、反向传播、更新参数。

结构图：

- [层结构图](tinycnn.svg)
- [Feature Map 堆叠图](tinycnn_feature_maps.svg)
- [逐步 Tensor 内容可视化](tensor_steps.html)

通用版查看器：[Metric Viewer](../../viewers/metric_viewer.html)

这个模板还应配套一个 `export_tensors.py`，用于导出 `tensor_data.json`，供通用查看器读取。

运行：

```powershell
python demos/template_000/train.py
```

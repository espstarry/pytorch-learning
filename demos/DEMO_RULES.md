# Demo 约定

以后每个模型 demo 都必须同时提供“能运行模型”和“能观察 tensor”两部分。

## 目录

```text
<model_name>/
  model.py
  train.py
  export_tensors.py
  tensor_data.json       # 运行 export_tensors.py 后生成
  README.md
```

## Tensor 导出要求

`export_tensors.py` 必须记录模型 forward 中最重要的步骤，至少包括：

- 输入
- 每个主要卷积、池化、循环或注意力模块的输出
- Flatten、Embedding、Linear 等形状变化明显的步骤
- 最终输出

每一步使用统一格式：

```json
{
  "name": "conv1",
  "shape": [1, 6, 28, 28],
  "values": [[/* batch 和 channel 的二维切片 */]],
  "description": "第一层卷积输出"
}
```

完整示例见 [`tensor_viewer/README.md`](tensor_viewer/README.md)。

## 实现方式

不要把 Tensor 导出逻辑混进模型结构。推荐在 `export_tensors.py` 中用 hook 或显式保存：

```python
steps.append({
    "name": "conv1",
    "shape": list(x.shape),
    "values": x.detach().cpu()[:2, :8].tolist(),
    "description": "第一层卷积输出",
})
```

导出后，用通用查看器打开：

```text
demos/tensor_viewer/viewer.html
```

这样每个新 demo 都有相同的学习入口：先看 `model.py`，再运行 `export_tensors.py`，最后在浏览器中逐步查看 tensor。

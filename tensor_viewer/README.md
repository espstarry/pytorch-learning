# Tensor Viewer

这是所有模型 demo 共用的 Tensor 查看器。它不负责运行模型，只负责把 PyTorch 导出的中间 tensor 变成可读的页面。

查看器支持：步骤导航、键盘切换、batch/channel 选择、热力图、数值表、鼠标悬停读值，以及当前二维切片的最小值、最大值、均值和标准差。

## 使用

1. 用 PyTorch 导出一个 JSON 文件。
2. 浏览器打开 `viewer.html`。
3. 把 JSON 文件拖到页面中，或点击选择文件。

JSON 格式：

```json
{
  "model": "LeNet",
  "steps": [
    {
      "name": "conv1",
      "shape": [1, 6, 28, 28],
      "values": [[/* batch=0, channel=0 的二维数组 */]],
      "description": "第一层卷积输出"
    }
  ]
}
```

为了控制文件大小，`values` 可以只保存部分 batch 和 channel；页面会按照数据实际提供的切片显示，并把完整 shape 单独展示出来。对于 `[batch, features]` 这样的二维 tensor，请把 `values` 保存为 `[batch, 1, features]`，让它在页面中显示为单行向量。

## 从 PyTorch 导出

```python
import json

saved = []

def save_step(name, x, description=""):
    saved.append({
        "name": name,
        "shape": list(x.shape),
        "values": x.detach().cpu()[:2, :8].tolist(),
        "description": description,
    })

# 在 forward 或 train.py 中：
# save_step("conv1", x, "第一层卷积输出")

with open("tensor_data.json", "w", encoding="utf-8") as f:
    json.dump({"model": "MyModel", "steps": saved}, f)
```

查看器会自动读取 `shape`、步骤名、二维切片和说明。

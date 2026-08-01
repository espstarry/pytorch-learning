# Loss Viewer

通用训练曲线查看器。打开 `viewer.html`，拖入如下 JSON：

```json
{
  "model": "LeNet",
  "curves": {
    "train_loss": [2.3, 1.8, 1.2],
    "test_loss": [2.2, 1.9, 1.5]
  }
}
```

也支持：

```json
{"model":"LeNet","points":[{"step":1,"train_loss":2.3},{"step":2,"train_loss":1.8}]}
```

import json
from pathlib import Path


class LossRecorder:
    """保存通用的训练曲线 JSON。"""

    def __init__(self, model="model"):
        self.model = model
        self.points = []

    def add(self, step, **values):
        self.points.append({"step": step, **values})

    def save(self, path="loss_data.json"):
        Path(path).write_text(
            json.dumps({"model": self.model, "points": self.points}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

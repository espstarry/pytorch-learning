import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("runs/.matplotlib").resolve()))

import matplotlib.pyplot as plt


RUNS_ROOT = Path("runs")
COMPARISON_CSV_PATH = RUNS_ROOT / "comparison.csv"
COMPARISON_JSON_PATH = RUNS_ROOT / "comparison.json"
ANALYSIS_JSON_PATH = RUNS_ROOT / "analysis.json"
METRICS_PLOT_PATH = RUNS_ROOT / "comparison_metrics.png"
CURVES_PLOT_PATH = RUNS_ROOT / "comparison_curves.png"


def load_json(path):
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def collect_runs(runs_root):
    """Collect one flat row plus the source directory for each completed run."""
    rows = []
    for summary_path in sorted(runs_root.glob("*/*/summary.json")):
        run_dir = summary_path.parent
        config = load_json(run_dir / "config.json")
        environment = load_json(run_dir / "environment.json")
        summary = load_json(summary_path)
        rows.append(
            {
                "run_dir": str(run_dir),
                "label": f"{config['experiment_name']}\n{run_dir.name}",
                **config,
                "git_commit": environment.get("git_commit"),
                "git_dirty": environment.get("git_dirty"),
                "pytorch": environment.get("pytorch"),
                **summary,
            }
        )
    return rows


def save_comparison_table(rows):
    """Save both spreadsheet-friendly CSV and structured JSON."""
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with COMPARISON_CSV_PATH.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with COMPARISON_JSON_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(rows, output_file, indent=2)


def save_analysis(rows):
    """Save simple rankings that are useful when reviewing experiments."""
    best_loss = min(rows, key=lambda row: row["best_valid_loss"])
    best_accuracy = max(rows, key=lambda row: row["best_valid_accuracy"])
    analysis = {
        "run_count": len(rows),
        "best_by_valid_loss": {
            "run_dir": best_loss["run_dir"],
            "value": best_loss["best_valid_loss"],
        },
        "best_by_valid_accuracy": {
            "run_dir": best_accuracy["run_dir"],
            "value": best_accuracy["best_valid_accuracy"],
        },
    }
    with ANALYSIS_JSON_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(analysis, output_file, indent=2)
    return analysis


def plot_metric_comparison(rows):
    labels = [row["label"] for row in rows]
    losses = [row["best_valid_loss"] for row in rows]
    accuracies = [row["best_valid_accuracy"] for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(labels, losses, color="#d62728")
    axes[0].set_title("Best validation loss")
    axes[0].set_ylabel("loss")
    axes[1].bar(labels, accuracies, color="#2ca02c")
    axes[1].set_title("Best validation accuracy")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.05)
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(METRICS_PLOT_PATH)
    plt.close(figure)


def plot_learning_curves(rows):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for row in rows:
        history = load_json(Path(row["run_dir"]) / "history.json")
        epochs = [item["epoch"] for item in history]
        valid_losses = [item["valid_loss"] for item in history]
        valid_accuracies = [item["valid_accuracy"] for item in history]
        axes[0].plot(epochs, valid_losses, label=row["label"])
        axes[1].plot(epochs, valid_accuracies, label=row["label"])

    axes[0].set_title("Validation loss by run")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[1].set_title("Validation accuracy by run")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[0].legend(fontsize="small")
    axes[1].legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(CURVES_PLOT_PATH)
    plt.close(figure)


def main():
    rows = collect_runs(RUNS_ROOT)
    if not rows:
        print("No completed runs found. Run an experiment first.")
        return

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    save_comparison_table(rows)
    analysis = save_analysis(rows)
    plot_metric_comparison(rows)
    plot_learning_curves(rows)

    print(f"compared {len(rows)} runs")
    print(f"best by validation loss: {analysis['best_by_valid_loss']['run_dir']}")
    print(f"best by validation accuracy: {analysis['best_by_valid_accuracy']['run_dir']}")
    print(f"comparison table saved at: {COMPARISON_CSV_PATH}")
    print(f"analysis saved at: {ANALYSIS_JSON_PATH}")
    print(f"plots saved at: {METRICS_PLOT_PATH}, {CURVES_PLOT_PATH}")


if __name__ == "__main__":
    main()

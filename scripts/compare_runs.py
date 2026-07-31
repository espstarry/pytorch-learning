import csv
import json
from pathlib import Path


RUNS_ROOT = Path("runs")
OUTPUT_PATH = RUNS_ROOT / "comparison.csv"


def load_json(path):
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def collect_runs(runs_root):
    rows = []
    for summary_path in sorted(runs_root.glob("*/*/summary.json")):
        run_dir = summary_path.parent
        config = load_json(run_dir / "config.json")
        environment = load_json(run_dir / "environment.json")
        summary = load_json(summary_path)
        rows.append(
            {
                "run_dir": str(run_dir),
                **config,
                "git_commit": environment.get("git_commit"),
                "git_dirty": environment.get("git_dirty"),
                "pytorch": environment.get("pytorch"),
                **summary,
            }
        )
    return rows


def save_comparison(rows, output_path):
    if not rows:
        print("No completed runs found.")
        return

    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with output_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"compared {len(rows)} runs")
    print(f"comparison saved at: {output_path}")


def main():
    save_comparison(collect_runs(RUNS_ROOT), OUTPUT_PATH)


if __name__ == "__main__":
    main()

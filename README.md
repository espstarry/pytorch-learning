# PyTorch Learning

This repository tracks small PyTorch exercises.

The goal is to learn by reading, filling in, running, and committing small examples.

## Current Exercise

- `exercises/001_fill_in_linear.py`

Goal:
Learn a linear function:

```text
y = 2x + 1
```

Core loop:

```text
forward -> loss -> backward -> update parameters -> clear gradients
```

Important PyTorch side effects:

- `loss.backward()` fills or accumulates `w.grad` and `b.grad`.
- `with torch.no_grad()` prevents the parameter update from becoming part of the computation graph.
- `w.grad.zero_()` and `b.grad.zero_()` clear gradients in-place.

## Suggested Workflow

```powershell
git status
git diff
python .\exercises\001_fill_in_linear.py
git add exercises\001_fill_in_linear.py
git commit -m "Complete first PyTorch linear exercise"
```

Before committing, always read `git diff` so the commit contains only what you intended.

## Reproducible Experiment Workflow

Exercise 011 uses a reusable experiment runner. Change only the values in
`ExperimentConfig`, then run:

```powershell
python -m exercises.011_refactored_classification
```

Each run is saved separately under:

```text
runs/<experiment_name>/<timestamp>/
```

The directory records the experiment configuration, Python/PyTorch/Git
environment, epoch history, summary, plots, best model, and latest checkpoint.

To combine all completed runs into a table for comparison:

```powershell
python -m scripts.compare_runs
```

The comparison tool produces:

```text
runs/comparison.csv          # spreadsheet-friendly table
runs/comparison.json         # structured data for Python analysis
runs/analysis.json           # best run by loss and accuracy
runs/comparison_metrics.png  # best metric bars
runs/comparison_curves.png   # validation curves across runs
```

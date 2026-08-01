# Project Instructions

This is a minimal PyTorch model-reading project, not an experiment framework.

For every new model demo:

- keep `model.py` short and focused on the architecture;
- keep `train.py` as a minimal forward/loss/backward/update example;
- add `export_tensors.py` to export key forward tensors to `tensor_data.json`;
- record each important shape-changing step, including input and output;
- make the JSON compatible with `demos/tensor_viewer/viewer.html`;
- use generated or tiny data and avoid metrics, logging, checkpoints, complex loaders, and experiment abstractions.

Before finishing a demo, check that the exported JSON can be opened in the Tensor Viewer and that the README links to it.

## Running Code

Do not automatically run training scripts, download datasets, or start long validations after writing a demo. First tell the user the exact command, what it will do, and an expected runtime range. Run code only when the user explicitly asks to run, test, or verify it.

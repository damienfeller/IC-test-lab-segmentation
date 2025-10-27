# IC Test Lab Segmentation

This repository provides lightweight utilities to configure nnU-Net experiments,
launch training runs, and perform inference for segmentation research.

## Package layout

All code lives under the `src/segmentation_app` package:

* `training/` – Experiment configuration, split management, trainer orchestration
  and CLI entry points (`python -m segmentation_app.training.run`).
* `inference/` – Utilities to run inference from trained checkpoints and export
  quality assurance metrics.

## Usage

Ensure the `src/` directory is available on your `PYTHONPATH` and that nnU-Net
is installed in the active Python environment.

### Training and evaluation

```bash
PYTHONPATH=src python -m segmentation_app.training.run train path/to/config.yaml
PYTHONPATH=src python -m segmentation_app.training.run eval path/to/config.yaml --checkpoint fold_0.ckpt
```

Configuration files are parsed into a `TrainerConfig` dataclass, ensuring that
experiment parameters can be versioned and reproduced across runs.

### Inference utilities

The `InferenceEngine` class bundles preprocessing, inference and postprocessing
for saved checkpoints and exports quality metrics in JSON format.

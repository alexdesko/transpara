# TRANSPARA

## TODO
- On the resnet keep 3 channels and rewrite the FC
- Remove the train script
- Check dependencies
- Implement gradCAM for the other models
- Check why the model is unbalanced
- ensure testing images recieve the same transformation as the train and val
- rewrite the docstrings
- Push
- Push to Github once everything is completed

Explainable AI training project for chest X‑ray pneumonia classification. Includes simple CNN/MLP baselines and a ResNet18 option, a training loop with progress, class imbalance handling, Hydra configs, and a Streamlit app for training, inference, explainability, and run browsing.

## Overview
- Models: `SimpleCNN`, `SimpleMLP`, `ResNet18` (1‑channel input)
- Trainer: supervised training with accuracy/loss tracking and CSV export
- Imbalance: weighted sampler and weighted cross‑entropy (configurable)
- Hardware: auto‑selects `mps` (Apple), `cuda`, or `cpu`
- Notebooks: exploratory analysis and training report
- Streamlit: train models, upload & predict, Grad‑CAM, and view runs

## Setup
- Python: 3.10+ recommended
- Create a virtual environment and install dependencies:
  - macOS/Windows/Linux (CPU example):
    - `python -m venv .venv && source .venv/bin/activate` (Windows: `\.venv\Scripts\activate`)
    - `pip install --upgrade pip`
  - `pip install torch torchvision pandas pillow pyyaml rich`
  - For GPU/MPS, install the appropriate `torch/torchvision` wheels per the official PyTorch instructions.

## Data
- Expected layout (relative to a chosen dataset root):
  - `<root>/train/NORMAL/*.(jpeg|jpg|png)`
  - `<root>/train/PNEUMONIA/*.(jpeg|jpg|png)`
  - `<root>/val/NORMAL/*.(jpeg|jpg|png)`
  - `<root>/val/PNEUMONIA/*.(jpeg|jpg|png)`
- Images are loaded as grayscale and normalized.
- Configure the dataset root either in Hydra config (`configs/data/chest_xray.yaml`) or via the Streamlit “Train Model” page.
- The dataset loader resolves relative paths from the repository root and validates the folder structure.

## Configuration (Hydra)
- Main entry: `configs/train.yaml` with defaults for model/data/training/optimizer/criterion/dataloader.
- Examples:
  - `configs/model/{cnn,mlp,resnet18}.yaml`
  - `configs/data/chest_xray.yaml`
  - `configs/training/base.yaml`
  - `configs/optimizer/adam.yaml`
  - `configs/criterion/{ce,weighted_ce}.yaml`
- Key options:
  - Model: `CNN` | `MLP` | `ResNet18` (+ `hidden_size` for MLP)
  - Data: `input_size`, `root`
  - Training: `batch_size`, `num_epochs`, `use_weighted_sampler`, `use_amp`, `seed`, `metric_to_monitor`, `mode`
  - Optimizer: `Adam` with `lr`
  - Dataloader: `num_workers`, `pin_memory`, `persistent_workers`

## Train (CLI)
- Run: `python scripts/train.py`
- Behavior:
  - Uses Hydra; outputs saved to `outputs/YYYY-MM-DD/HH-MM-SS_<experiment_name>/`.
  - Artifacts: `config.yaml`, `model.pth`, `best_model.pth`, `metrics.csv`.
  - Weighted sampler and weighted cross‑entropy are configurable.

## Notebooks
- `notebooks/exploratory_data_analysis.ipynb`: quick EDA of the dataset
- `notebooks/training_report.ipynb`: visualize metrics after training

## Streamlit App
- Run the demo UI locally:
  - `pip install -r streamlit_app/requirements.txt` (or rely on your env)
  - `streamlit run streamlit_app/app.py`
- Pages:
  - Train Model: start a training run with UI‑controlled parameters (overrides Hydra defaults). Shows live epoch progress, loss/accuracy charts, and saves artifacts to the same `outputs/...` structure.
  - Try Model: upload an image → predictions using a selected run’s `model.pth` and `config.yaml`.
  - Explainability: Grad‑CAM heatmap for ResNet18 models.
  - Training Runs: browse `outputs/` or `trained_models/` metrics and configs; view charts.
- Notes:
  - The Streamlit page saves YAML configs with Enums coerced to plain strings for compatibility.
  - `persistent_workers` is enabled only if `num_workers > 0`.

## Roadmap
- Add early stopping and learning‑rate schedulers.
- Resume training from a checkpoint in the UI.
- Extend explainability beyond ResNet18.

## Notes
- Dataset loader accepts `.jpeg`, `.jpg`, and `.png`.
- `SimpleCNN` and `SimpleMLP` are minimal baselines for iteration.

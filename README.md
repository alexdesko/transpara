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

Explainable AI training project for chest X‑ray pneumonia classification. Uses a ResNet18 model, a training loop with progress, class imbalance handling, a single-file Hydra config, and a Streamlit app for training, inference, explainability, and run browsing.

## Overview
- Model: `ResNet18`
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
- Expected layout (single root with class folders):
  - `<root>/<CLASS>/*.(jpeg|jpg|png)` for each class (e.g., `COVID`, `NORMAL`, `PNEUMONIA`)
- The loader deterministically splits each class 80/10/10 into train/val/test. If you already have `<root>/{train,val,test}/<CLASS>/...`, that layout is supported too.
- Configure the dataset root in `configs/train.yaml` or via the Streamlit “Train Model” page.

## Configuration (Hydra)
- Single file: `configs/train.yaml` containing `model`, `data`, `training`, `optimizer`, `criterion`, and `dataloader` sections.
- Key options:
  - Model: `name`=`ResNet18`, `num_classes`
  - Data: `root`, `input_size`, `split_seed`
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

# TRANSPARA

## TODO
- Remove the train script
- Check dependencies
- Implement gradCAM with the proper library
- rewrite the docstrings
- Push to Github once everything is completed

Explainable AI project for chest X‑ray pneumonia classification with a ResNet18 backbone, a lightweight Trainer, Hydra configuration for CLI runs, and a Streamlit app for training, inference, and Grad‑CAM explainability.

## Overview
- Model: pretrained `ResNet18`; backbone frozen, final FC fine‑tuned
- Trainer: supervised loop with progress and CSV metrics export
- Devices: Streamlit training auto‑selects `mps`/`cuda`/`cpu`; CLI currently uses CPU
- Config: single Hydra file `configs/train.yaml` drives CLI runs
- Streamlit: pages for training, trying a model, and explainability (Grad‑CAM)
- Notebooks: exploratory data analysis (`notebooks/exploratory_data_analysis.ipynb`)

## Quickstart
- Python: 3.10+ recommended
- Create and activate a virtual environment, then install core libs:
  - `python -m venv .venv && source .venv/bin/activate` (Windows: `\.venv\Scripts\activate`)
  - `pip install --upgrade pip`
  - `pip install -e .`  # installs from `pyproject.toml`
- For the Streamlit app (UI):
  - `pip install -r streamlit_app/requirements.txt`

## Data
- Expected layout (single root with class folders, PNG only):
  - `dataset/NORMAL/*.png`
  - `dataset/PNEUMONIA/*.png`
  - `dataset/COVID/*.png`
- Splitting: deterministic 80/10/10 per class by contiguous slicing. Class label mapping: `NORMAL=0`, `PNEUMONIA=1`, `COVID=2`.
- Debug mode: when `DEBUG=1` (see `.env`), the training split is subsampled to ~10% to speed up iterations.

## Configuration (Hydra)
- Single file: `configs/train.yaml` with sections: `model`, `data`, `training`, `optimizer`, `criterion`, and Hydra’s `run.dir` pattern for outputs.
- Key options used by the code:
  - Model: `model.name`=`ResNet18`, `model.num_classes`
  - Data: `data.root` (dataset path)
  - Training: `training.batch_size` (Streamlit), `training.num_epochs`, `training.seed`, `training.metric_to_monitor`, `training.mode`
  - Optimizer: `optimizer.lr`, `optimizer.wd`
  - Criterion: `criterion.name` (e.g., `CrossEntropyLoss`)

## Train (CLI)
- Run: `python scripts/train.py`
- Behavior:
  - Uses Hydra to load `configs/train.yaml` and construct the model and training loop.
  - Artifacts: `config.yaml`, `model.pth`, `best_model.pth`, `metrics.csv`.
  - Output directory: by default under `outputs/YYYY-MM-DD/HH-MM-SS_<experiment_name>/` (Hydra). When `DEBUG=1`, artifacts are written to `./temp_DEBUG/` and epochs are capped for faster iteration.
  - Note: current CLI path runs on CPU; Streamlit training selects the best available device.

## Streamlit App
- Launch: `streamlit run streamlit_app/app.py`
- Pages:
  - Train Model (`streamlit_app/pages/0_Train_Model.py`): configure dataset path, batch size, epochs, LR; runs training with live charts and saves artifacts under `outputs/`.
  - Try Model (`streamlit_app/pages/1_Try_Model.py`): upload an image and view class probabilities from a selected run.
  - Explainability (`streamlit_app/pages/2_Explainability.py`): Grad‑CAM heatmaps for ResNet18 runs.
- Runs discovery: the app looks for runs under `outputs/` and `trained_models/` that contain `metrics.csv`.

## Repository Structure
- `scripts/train.py`: CLI entry that delegates to `src/trainer/launch.py` (Hydra)
- `configs/train.yaml`: single Hydra config for CLI training
- `src/models/resnet.py`: `CustomResNet18` (pretrained, frozen backbone, replaced FC)
- `src/trainer/trainer.py`: minimal training/validation loop + metric/export helpers
- `src/trainer/launch.py`: Hydra launcher, wiring data/model/optimizer/criterion
- `src/dataio`: dataset split utilities and transforms
- `streamlit_app/`: Streamlit UI, pages, and shared app utilities

## Notes
- File formats: current dataset utility loads `*.png` files only.
- Transform parity: both training and inference use the same ResNet18 default transforms.
- Devices: `src/utils/device.py` selects `mps` → `cuda` → `cpu` for Streamlit training.

## Roadmap
- Early stopping and learning‑rate schedulers
- Resume training from a checkpoint in the UI
- Extend explainability beyond ResNet18

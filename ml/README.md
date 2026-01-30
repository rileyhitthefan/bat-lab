# BatLab ML (refactored)

This repo is a refactor-friendly structure for your BatLab model.

## Quickstart (VS Code / terminal)

1) Create and activate a virtual environment
- macOS/Linux:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
- Windows:
  - `python -m venv .venv`
  - `.\.venv\Scripts\activate`

2) Install deps
- `pip install -r requirements.txt`

3) Train
- `python scripts/train.py --data-root "PATH/TO/Bat Calls"`

4) Predict (single wav)
- `python scripts/predict.py --wav "PATH/FILE.wav" --location "unknown" --ckpt models/checkpoints/best_model.pt --data-root "PATH/TO/Bat Calls"`

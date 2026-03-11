# Bat Lab

Detect and classify bat echolocation calls and identify behavioral sequences.

- **Demo:** https://drive.google.com/file/d/1GSTknpda1oaCgl3L0ylzwszFyxQ6K8qM/view?usp=sharing
- **User Manual:** https://drive.google.com/drive/folders/10A_5EJbl0U-VE_eGgLaygLFI4HIdAO14?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto

## Features

- **Classify** — Run inference on a folder of `.wav` files. Choose the **Colab (base)** model or any **subset model** trained in the app.
- **Add Detector** — Register detector locations (stored in MySQL).
- **Add Species** — Register species (stored in MySQL).
- **Add Training Data** — Upload training calls per detector/species; files are stored in the DB (Call_Library).
- **Train New Model** — Fine-tune from the Colab base model on selected detectors and species. Subset models are saved under `model_checkpoints/local/<subset_name>/` and appear in the Classify model dropdown.

## Layout
```
bat-lab/
├── app.py                              # Streamlit UI
├── configs/
│   └── default.yaml                    # Mel/training config (sr, n_mels, paths, etc.)
├── model_checkpoints/
│   ├── colab/                          # Base model (for fine-tuning and default inference)
│   │   ├── best_model.pt
│   │   ├── feature_stats.json
│   │   ├── data_manifest.csv
│   │   └── thresholds.yaml
│   └── local/                          # Subset models (one folder per run)
│       └── <subset_name>/
│           ├── <subset_name>.pt
│           ├── <subset_name>_metadata.json
│           └── feature_stats.json
├── src/
│   ├── batlab.sql                      # MySQL schema
│   ├── db/
│   │   ├── connection.py               # High-level read/write (species, detectors, Call_Library)
│   │   └── __init__.py
│   ├── ml/
│   │   ├── config.py                  # Config + paths
│   │   ├── classify_app.py            # Classification for Streamlit
│   │   ├── data/                      # Manifest, dataset, features, spectrograms
│   │   ├── models/                    # BatClassifier (net.py)
│   │   ├── scripts/
│   │   │   ├── train.py               # CLI: build manifest + full train
│   │   │   └── predict.py             # Single-file inference
│   │   └── training/
│   │       ├── inference.py            # load_best, predict_file
│   │       ├── calibration.py          # Temperature scaling
│   │       ├── trainer.py              # Training loop
│   │       └── subset_model_trainer.py # Fine-tune subset from UI (colab → local)
│   └── ui/
│       ├── styles.css                  # Base (light) styles
│       └── theme_dark.css              # Dark theme overrides
├── cache_mels/                         # Cached mel spectrograms (optional, gitignored)
├── cache_numfeats/                     # Cached numeric features (optional, gitignored)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Setup

From the repo root:

```bash
cd bat-lab
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
pip install -e .
```

**MySQL** is required. Apply the schema and set connection params:

- Schema: `src/batlab.sql`
- Env: `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`

Put your bat-call `.wav` data where you like; the UI uses the DB Call_Library (from **Add Training Data**) for subset training. For CLI training you pass `--data-root` to the train script.

## Run

**Application shortcut (no code):** `batlabexe.bat`

- **UI:**  
  `streamlit run app.py`

- **CLI — full training** (builds manifest, trains from scratch, writes to `model_checkpoints/`):  
  ```bash
  python -m src.ml.scripts.train --data-root path/to/bat/calls
  ```

- **CLI — train with options:**  
  ```bash
  python -m src.ml.scripts.train \
    --data-root path/to/bat/calls \
    --config configs/default.yaml \
    --min-samples 5 \
    --test-size 0.3 \
    --val-size 0.5
  ```

- **CLI — single-file prediction:**  
  ```bash
  python -m src.ml.scripts.predict \
    --wav path/to/file.wav \
    --ckpt model_checkpoints/colab/best_model.pt \
    --data-root path/to/root \
    --config configs/default.yaml
  ```

After install you can use: `batlab-train` and `batlab-predict` (same arguments).

## Models

| Path | Description |
|------|-------------|
| `model_checkpoints/colab/best_model.pt` | Base model; used as the starting point for subset fine-tuning and as the default in the Classify dropdown. |
| `model_checkpoints/local/<subset_name>/<subset_name>.pt` | Subset models produced by **Train New Model** in the app; selectable in the Classify tab. |

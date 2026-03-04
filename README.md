# Bat Lab

Detect and classify bat echolocation calls and identify behavioral sequences.

## Layout

```
bat-lab/
├── app.py                         # Streamlit UI
├── configs/
│   └── default.yaml               # Mel/training config (sr, n_mels, paths, etc.)
├── model_checkpoints/             # Output: manifest, checkpoints, thresholds (gitignored in practice)
│   ├── data_manifest.csv          # Built by train script (filepath, label, location)
│   ├── feature_stats.json
│   ├── thresholds.yaml
│   └── best_model.pt
├── src/
│   ├── batlab.sql                 # MySQL schema
│   ├── db/                        # DB connection and helpers
│   │   ├── connection.py          # High-level read/write (species, detectors, training data)
│   │   └── mysql_connection.py   # Low-level connection
│   ├── ml/                        # ML package
│   │   ├── config.py              # Config + paths
│   │   ├── classify_app.py        # Classification for Streamlit
│   │   ├── data/                  # Manifest, dataset, features, spectrograms
│   │   ├── models/                # BatClassifier (net.py)
│   │   ├── scripts/               # CLI entry points
│   │   │   ├── train.py           # Build manifest + train
│   │   │   └── predict.py         # Single-file inference
│   │   ├── training/              # Trainer, inference, calibration
│   │   └── utils/
│   └── ui/                        # Styles for Streamlit
├── cache_mels/                    # Cached mel spectrograms (optional, gitignored)
├── cache_numfeats/                # Cached numeric features (optional, gitignored)
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

MySQL is optional; the app and training work without it. If you use the DB, ensure the schema is applied (`src/batlab.sql`) and connection params are configured.

`data/`, `cache_mels/`, and `cache_numfeats/` are gitignored; put your bat-call `.wav` data under a directory of your choice and pass it as `--data-root`.

## Run

Application shortcut (no code) : `test.bat`

- **UI:**  
  `streamlit run app.py`

- **Build manifest and train** (manifest is written to `model_checkpoints/data_manifest.csv` by default):  
  `python -m src.ml.scripts.train --data-root path/to/bat/calls`

- **Train using an existing manifest:**  
  `python -m src.ml.scripts.train --data-root path/to/bat/calls --manifest model_checkpoints/data_manifest.csv`

- **Train with config and options:**  
  ```bash
  python -m src.ml.scripts.train \
    --data-root path/to/bat/calls \
    --config configs/default.yaml \
    --min-samples 5 \
    --test-size 0.3 \
    --val-size 0.5
  ```

- **Single-file prediction:**  
  ```bash
  python -m src.ml.scripts.predict \
    --wav path/to/file.wav \
    --ckpt model_checkpoints/best_model.pt \
    --data-root path/to/root \
    --config configs/default.yaml
  ```

After install you can also use: `batlab-train` and `batlab-predict` (same arguments).

# Bat Lab

Detect and classify bat echolocation calls and identify behavioral sequences.

## Layout

```
bat-lab/
├── app.py                    # Streamlit UI
├── src/
│   ├── ml/                   # ML package (config, data, models, training, utils)
│   ├── classifier/           # SmallAudioCNN + .pt checkpoints
│   ├── db/                   # DB connection (connection.py)
│   ├── sounduploader/        # Upload, sonogram generation
│   └── batlab.sql            # Schema (MySQL)
├── configs/
│   └── default.yaml          # Mel/training config
├── scripts/
│   ├── predict.py            # Single-file inference
│   └── train.py              # Training (needs ml.data.manifest, dataset)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Setup

```bash
cd bat-lab
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -e .
pip install -r requirements.txt
```

## Run

- **UI:** `streamlit run app.py`
- **Predict:** `python scripts/predict.py --wav path/to/file.wav --ckpt models/checkpoints/best_model.pt --data-root path/to/data --config configs/default.yaml --thresholds thresholds.yaml`
- **Train:** `python scripts/train.py --data-root path/to/Bat\ Calls` (requires manifest/dataset modules)

## Colab

https://colab.research.google.com/drive/10E7C46wI7LC_fU7LjXwGmOvYGLCOuaM4?usp=sharing

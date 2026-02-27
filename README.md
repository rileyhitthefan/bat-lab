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
- **Build manifest** `python -m src.ml.scripts.train --data-root path/to/bat/calls`
- **Train** `python -m src.ml.scripts.train --data-root path/to/bat/calls --manifest scripts/data_manifest.csv`
- **Train with configs**:
  `python -m src.ml.scripts.train \
    --data-root path/to/bat/calls \
    --config configs/default.yaml \
    --min-samples 5 \
    --test-size 0.3 \
    --val-size 0.5`

## Colab

https://colab.research.google.com/drive/1cX76dSk2g4pe60X0_PX_W5D2iN44MJLj?usp=sharing

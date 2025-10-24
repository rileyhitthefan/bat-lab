# Bat Lab 

Code for detecting and classifying bat echolocation calls and identifying behavioral sequences

bat-lab/
├── app.py                          # UI entry point
│
├── src/
│   ├── __init__.py
│   │
│   ├── sounduploader/
│   │   ├── __init__.py
│   │   ├── file_upload.py          # Upload + validation
│   │   └── sonogram.py             # Generate spectrograms
│   │
│   ├── classifier/
│   │   ├── __init__.py
│   │   └── model.py                # ML logic
│   │
│   └── static/                     
│       ├── uploads/                # Temporary uploaded .wav files
│       ├── results/                # Model output files
│       └── data/                   # Reference/training data
│
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore


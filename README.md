# Bat Lab 

Code for detecting and classifying bat echolocation calls and identifying behavioral sequences

```
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
│   │   ├── uploads/                # Temporary uploaded .wav files
│   │   ├── results/                # Model output files
│   │   └── data/                  # Reference/training data
│   │
│   └── db/
│       ├── __init__.py
│       ├── connection.py      # Database connection setup
│       └── queries.py         # DB access endpoints
│
├── config/
│   ├── __init__.py
│   └── settings.py            # Loads env vars, MySQL credentials
│
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

https://colab.research.google.com/drive/10E7C46wI7LC_fU7LjXwGmOvYGLCOuaM4?usp=sharing

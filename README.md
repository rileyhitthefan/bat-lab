# Bat Lab 

Code for detecting and classifying bat echolocation calls and identifying behavioral sequences
- Demo: https://drive.google.com/file/d/1GSTknpda1oaCgl3L0ylzwszFyxQ6K8qM/view?usp=sharing

File for User Manual: https://drive.google.com/drive/folders/10A_5EJbl0U-VE_eGgLaygLFI4HIdAO14?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto 
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

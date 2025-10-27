import os
import shutil

def is_wav(filename: str) -> bool:
    # Check if the uploaded file has a .wav extension.
    return filename.lower().endswith(".wav")

def save_temp_file(file, upload_session, upload_dir: str = "./src/static/uploads") -> str:
    # Save uploaded file to the temporary upload directory.
    temp_dir = os.path.join(upload_dir, upload_session)
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def clear_temp_dir():
    # Clean up temporary storage directory.
    temp_dir = os.path.join(os.path.dirname(__file__), "temp_storage")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

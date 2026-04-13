"""
app.py — BatLab: Bat Acoustic Identification Application
==========================================================
A Streamlit-based web application that allows researchers to:
  1. Classify bat acoustic .wav files using a (simulated) ML model.
  2. Register and manage acoustic detectors (hardware units in the field).
  3. Register and manage bat species records.
  4. Upload training audio data linked to a species + detector.
  5. Kick off new model training runs over selected detectors/species.

The app requires a simple username/password login before the main UI is shown.
All state (detectors, species, training data, classification results) is held
in Streamlit's session_state, so it persists across widget interactions within
a single browser session but is reset when the page is refreshed.
"""

import streamlit as st
import pandas as pd
import time
from pathlib import Path
import io
import contextlib
import shutil
import sys

from src.ml.classify_app import classify_uploaded_files
from src.ml.config import Config
from src.db.connection import (
    get_call_library_data,
    save_detectors,
    save_species,
    save_training_data,
    update_call_library_data,
    delete_detectors,
    delete_species,
    delete_call_library_data,
    load_detectors_from_db,
    load_species_from_db,
    load_training_records_df,
)
from src.ml.training.subset_model_trainer import create_subset_model_from_ui_selection
from src.ml.training.full_model_trainer import retrain_full_model_from_ui

# ============================================================================
# MODEL DISCOVERY (for Classify tab dropdown)
# ============================================================================

def get_available_models() -> list[tuple[str, str]]:
    """
    Return list of (display_label, model_path_str) for the Classify tab:
    Priority order:
    - Full retrained models under model_checkpoints/full_retrained/<name>/<name>.pt
    - Colab (base) model at model_checkpoints/colab/best_model.pt
    - All subset models under model_checkpoints/local/<name>/<name>.pt
    """
    project_root = Path(__file__).resolve().parent
    cfg = Config.from_yaml(project_root / "configs" / "default.yaml")
    base = project_root / cfg.model_dir
    options: list[tuple[str, str]] = []

    # 1) Full retrained models (priority)
    full_dir = base / "full_retrained"
    if full_dir.is_dir():
        full_entries: list[tuple[float, str, str]] = []
        for subdir in full_dir.iterdir():
            if not subdir.is_dir():
                continue
            pt = subdir / f"{subdir.name}.pt"
            if not pt.exists():
                continue
            try:
                sort_key = pt.stat().st_mtime
            except Exception:
                sort_key = 0.0
            full_entries.append((sort_key, subdir.name, str(pt)))

        # Newest first
        for _mtime, name, path_str in sorted(full_entries, key=lambda t: t[0], reverse=True):
            options.append((f"Full retrained: {name}", path_str))

    # 2) Colab base model
    colab_pt = base / "colab" / "best_model.pt"
    if colab_pt.exists():
        options.append(("Colab (base model)", str(colab_pt)))

    # 3) Subset models
    local_dir = base / "local"
    if local_dir.is_dir():
        for subdir in sorted(local_dir.iterdir()):
            if subdir.is_dir():
                subset_pt = subdir / f"{subdir.name}.pt"
                if subset_pt.exists():
                    options.append((f"Subset: {subdir.name}", str(subset_pt)))

    return options

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(page_title="Bat Acoustic Identification", layout="wide")


def inject_css(filename: str) -> None:
    """Load a CSS file from src/ui/ and inject it into the page."""
    path = Path(__file__).resolve().parent / "src" / "ui" / filename
    if path.exists():
        st.markdown(f"<style>\n{path.read_text(encoding='utf-8')}\n</style>", unsafe_allow_html=True)

# Inject base (light) styles
inject_css("styles.css")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'known_data' not in st.session_state:
    st.session_state.known_data = pd.DataFrame(columns=['Filename', 'Species Prediction', 'Confidence Level'])

if 'unknown_data' not in st.session_state:
    st.session_state.unknown_data = pd.DataFrame(columns=['Filename'])

if 'export_data' not in st.session_state:
    st.session_state.export_data = pd.DataFrame(columns=[
        "IN FILE", "DATE", "TIME", "AUTO ID*", "MATCHING",
        "Fc", "Dur", "Fmax", "Fmin", "SC"
    ])

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'training_file_bytes' not in st.session_state:
    st.session_state.training_file_bytes = {}  # filename -> bytes (accumulated across submissions)

if 'detectors' not in st.session_state:
    detectors, det_errors = load_detectors_from_db()
    st.session_state.detectors = detectors
    if det_errors:
        for msg in det_errors:
            st.warning(f"Database warning (detectors): {msg}")

if 'species' not in st.session_state:
    # Load species strictly from the database (no manifest placeholders)
    species_from_db, sp_errors = load_species_from_db()
    if sp_errors:
        for msg in sp_errors:
            st.warning(f"Database warning (species): {msg}")
    st.session_state.species = species_from_db or []

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'training_uploader_key' not in st.session_state:
    st.session_state.training_uploader_key = 0

if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0

if 'source_folder' not in st.session_state:
    st.session_state.source_folder = ""

if 'show_id_folder_form' not in st.session_state:
    st.session_state.show_id_folder_form = False

if 'show_unk_folder_form' not in st.session_state:
    st.session_state.show_unk_folder_form = False

if 'id_folder_result' not in st.session_state:
    st.session_state.id_folder_result = None

if 'unk_folder_result' not in st.session_state:
    st.session_state.unk_folder_result = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _file_like_from_disk(source_folder: str, filename: str):
    """Build a file-like object (with .name and .getvalue()) for classify_uploaded_files."""
    path = Path(source_folder) / filename
    if not path.is_file():
        return None

    class FileLike:
        def __init__(self, name: str, path: Path):
            self.name = name
            self._path = path

        def getvalue(self):
            return self._path.read_bytes()

    return FileLike(filename, path)


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert dataframe to CSV bytes for Streamlit download."""
    return df.to_csv(index=False).encode("utf-8")


def process_audio_files(
    source_folder: str,
    wav_filenames: list[str],
    model_path: str | None = None,
):
    """
    Classify .wav files from the given folder using the ML model (classify_app).
    Returns (known_df, unknown_df, export_df).
    """
    file_objects = []
    missing_or_skipped = []

    for fn in wav_filenames:
        if not fn.lower().endswith(".wav"):
            missing_or_skipped.append({"Filename": fn})
            continue
        fl = _file_like_from_disk(source_folder, fn)
        if fl is not None:
            file_objects.append(fl)
        else:
            missing_or_skipped.append({"Filename": fn})

    known_df, unknown_df, export_df = classify_uploaded_files(
        file_objects,
        model_path=model_path
    )

    if missing_or_skipped:
        unknown_df = pd.concat(
            [unknown_df, pd.DataFrame(missing_or_skipped)],
            ignore_index=True,
        )

    return known_df, unknown_df, export_df


# ============================================================================
# MAIN UI LAYOUT
# ============================================================================

# ============================================================================
# LOGIN PAGE
# ============================================================================
if not st.session_state.logged_in:
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Display BatLab logo
        st.image("batlablogo.PNG", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #000000; font-weight: bold;'>Please Login</h2>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")

            login_button = st.form_submit_button("Login", use_container_width=True)

            if login_button:
                if username == "admin" and password == "password":
                    st.session_state.logged_in = True
                    st.success("Login successful!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Invalid login credentials. Please try again.")

    st.stop()


# ============================================================================
# MAIN APPLICATION (After Login)
# ============================================================================

if st.session_state.logged_in:
    inject_css("theme_dark.css")

# Header
st.title("Welcome to the BatLab!")
st.markdown("Analyze bat acoustic calls and identify species using machine learning")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Classify", "Add Detector", "Add Species", "Add Training Data", "Train New Model"])

# ============================================================================
# TAB 1: CLASSIFY
# ============================================================================
with tab1:
    st.markdown("---")
    st.header("Classify Bat Acoustic Calls")

    # ── Start New Session ────────────────────────────────────────────────────
    if (
        not st.session_state.known_data.empty
        or not st.session_state.unknown_data.empty
        or not st.session_state.export_data.empty
    ):
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("**Start New Session**", use_container_width=True, key="btn_start_new_session"):
                st.session_state.known_data = pd.DataFrame(columns=['Filename', 'Species Prediction', 'Confidence Level'])
                st.session_state.unknown_data = pd.DataFrame(columns=['Filename'])
                st.session_state.export_data = pd.DataFrame(columns=[
                    "IN FILE", "DATE", "TIME", "AUTO ID*", "MATCHING",
                    "Fc", "Dur", "Fmax", "Fmin", "SC"
                ])
                st.session_state.uploaded_files = []
                st.session_state.source_folder = ""
                st.session_state.show_id_folder_form = False
                st.session_state.show_unk_folder_form = False
                st.session_state.id_folder_result = None
                st.session_state.unk_folder_result = None
                st.session_state.file_uploader_key += 1
                st.success("✅ Session reset! Ready for new files.")
                st.rerun()

    st.markdown("---")

    # ── STEP 1: Model Selector ───────────────────────────────────────────────
    st.markdown("### Step 1 — Select a Model")
    model_options = get_available_models()
    if model_options:
        labels = [o[0] for o in model_options]
        label_to_path = {o[0]: o[1] for o in model_options}
        selected_label = st.selectbox(
            "Model",
            labels,
            key="classify_model_select",
            help="Colab (base) or a subset model trained in the Train New Model tab.",
        )
        selected_model_path = label_to_path.get(selected_label)
    else:
        selected_model_path = None
        st.warning(
            "No models found. Add model_checkpoints/colab/best_model.pt or train a subset model in the Train New Model tab."
        )

    st.markdown("---")

    # ── STEP 2: Source folder path ───────────────────────────────────────────
    st.markdown("### Step 2 — Enter the source folder path")

    import os
    source_input = st.text_input(
        "Source folder path *",
        value=st.session_state.source_folder,
        placeholder=r"e.g.  /Users/you/BatRecordings  or  C:\BatRecordings",
        key="source_folder_input"
    )
    if st.button("**Verify Path & Load Files**", use_container_width=True, key="btn_verify_path"):
        raw_input = (source_input or "").strip()
        if not raw_input:
            st.error("Please enter a folder path.")
        else:
            try:
                source_path = Path(raw_input).expanduser().resolve()
            except Exception as exc:
                st.error(f"Invalid path: {exc}")
                source_path = None

            if source_path is not None:
                if not source_path.is_dir():
                    st.error(f"Path not found or is not a folder: '{source_path}'")
                else:
                    wav_files = [f.name for f in source_path.iterdir()
                                 if f.is_file() and f.suffix.lower() == '.wav']
                    if not wav_files:
                        st.warning(f"No .wav files found in '{source_path}'.")
                    else:
                        normalised = str(source_path)
                        st.session_state.source_folder = normalised
                        st.session_state.uploaded_files = wav_files
                        st.session_state.known_data = pd.DataFrame(columns=['Filename', 'Species Prediction', 'Confidence Level'])
                        st.session_state.unknown_data = pd.DataFrame(columns=['Filename'])
                        st.session_state.export_data = pd.DataFrame(columns=[
                            "IN FILE", "DATE", "TIME", "AUTO ID*", "MATCHING",
                            "Fc", "Dur", "Fmax", "Fmin", "SC"
                        ])
                        st.session_state.id_folder_result = None
                        st.session_state.unk_folder_result = None
                        st.success(f"✅ Found {len(wav_files)} .wav file(s) in '{normalised}'.")
                        st.rerun()

    # Show current verified folder and file count
    if st.session_state.source_folder and st.session_state.uploaded_files:
        st.info(f"**Source:** `{st.session_state.source_folder}` — "
                f"**{len(st.session_state.uploaded_files)}** .wav file(s) ready for classification.")

    st.markdown("---")

    # ── STEP 3: Classify ─────────────────────────────────────────────────────
    st.markdown("### Step 3 — Classify")

    classify_disabled = not bool(st.session_state.source_folder and st.session_state.uploaded_files)

    with st.form("classify_form"):
        submitted = st.form_submit_button(
            "Classify",
            use_container_width=True,
            disabled=classify_disabled
        )

        if submitted and not classify_disabled:
            with st.spinner("Analyzing audio files… Please wait."):
                known_df, unknown_df, export_df = process_audio_files(
                    st.session_state.source_folder,
                    st.session_state.uploaded_files,
                    model_path=selected_model_path,
                )

                if not known_df.empty:
                    st.session_state.known_data = pd.concat(
                        [st.session_state.known_data, known_df], ignore_index=True
                    )
                if not unknown_df.empty:
                    st.session_state.unknown_data = pd.concat(
                        [st.session_state.unknown_data, unknown_df], ignore_index=True
                    )
                if not export_df.empty:
                    st.session_state.export_data = pd.concat(
                        [st.session_state.export_data, export_df], ignore_index=True
                    )

            st.success(f"✅ Classification complete! Processed {len(st.session_state.uploaded_files)} file(s).")
            st.rerun()

    st.markdown("---")

    # ── IDENTIFIED SPECIES TABLE + MOVE-TO-FOLDER ────────────────────────────
    if not st.session_state.known_data.empty:
        known_count = len(st.session_state.known_data)
        st.subheader(f"😄 Identified Species — {known_count} file{'s' if known_count != 1 else ''}")
        st.dataframe(
            st.session_state.known_data,
            use_container_width=True,
            hide_index=True,
            height=400,
            column_config={
                "Filename": st.column_config.TextColumn("Filename"),
                "Species Prediction": st.column_config.TextColumn("Species Prediction"),
                "Confidence Level": st.column_config.TextColumn("Confidence Level"),
            }
        )

        if st.button("**Create New Folder for Identified Sound Files**",
                     key="btn_id_folder", use_container_width=True):
            st.session_state.show_id_folder_form = not st.session_state.show_id_folder_form
            st.session_state.id_folder_result = None

        if st.session_state.show_id_folder_form:
            with st.form("id_folder_form"):
                st.markdown("**Enter the full path for the new identified species folder**")
                st.markdown(
                    "The folder will be created if it does not exist. Files will be **moved** "
                    "from the source folder into species subfolders (e.g. `Identified_Bats/Myotis lucifugus/`)."
                )
                id_folder_path = st.text_input(
                    "Destination folder path *",
                    placeholder=r"e.g.  /Users/you/Identified_Bats  or  C:\Identified_Bats"
                )
                col_save_id, col_cancel_id = st.columns(2)
                with col_save_id:
                    confirm_id = st.form_submit_button("**Create Folder & Move Files**", use_container_width=True)
                with col_cancel_id:
                    cancel_id = st.form_submit_button("**Cancel**", use_container_width=True)

                if cancel_id:
                    st.session_state.show_id_folder_form = False
                    st.rerun()

                if confirm_id:
                    import os, shutil
                    raw_id = (id_folder_path or "").strip()
                    if raw_id:
                        try:
                            id_folder_path = str(Path(raw_id).expanduser().resolve())
                        except Exception:
                            id_folder_path = raw_id
                    else:
                        id_folder_path = ""

                    if not id_folder_path:
                        st.error("Please enter a destination folder path.")
                    elif os.path.abspath(id_folder_path) == os.path.abspath(st.session_state.source_folder):
                        st.error("Destination must be different from the source folder.")
                    else:
                        known_names = st.session_state.known_data['Filename'].tolist()
                        prediction_lookup = dict(
                            zip(
                                st.session_state.known_data['Filename'],
                                st.session_state.known_data['Species Prediction'],
                            )
                        )
                        moved, failed = [], []
                        for fname in known_names:
                            src = os.path.join(st.session_state.source_folder, fname)
                            if not os.path.isfile(src):
                                failed.append(f"{fname} (not found in source)")
                                continue

                            species_code = prediction_lookup.get(fname, "Unknown_Species")
                            safe_species = species_code.replace("/", "-").replace("\\", "-").strip()
                            if not safe_species:
                                safe_species = "Unknown_Species"

                            species_folder = os.path.join(id_folder_path, safe_species)
                            try:
                                os.makedirs(species_folder, exist_ok=True)
                            except Exception as exc:
                                failed.append(f"{fname} (could not create subfolder: {exc})")
                                continue

                            dst = os.path.join(species_folder, fname)
                            try:
                                shutil.move(src, dst)
                                moved.append(f"{fname} → {safe_species}/")
                            except Exception as exc:
                                failed.append(f"{fname} ({exc})")

                        st.session_state.id_folder_result = {
                            "path": id_folder_path,
                            "moved": moved,
                            "failed": failed,
                        }
                        st.session_state.show_id_folder_form = False
                        st.rerun()

        if st.session_state.id_folder_result:
            res = st.session_state.id_folder_result
            if res["moved"]:
                st.success(
                    f"Moved **{len(res['moved'])}** file(s) into species subfolders under `{res['path']}`."
                )
            if res["failed"]:
                st.warning("The following files could not be moved:\n" +
                           "\n".join(f"• {f}" for f in res["failed"]))

    else:
        st.info("No identified species yet. Verify a source folder and run classify to get started!")

    st.markdown("---")

    # ── UNKNOWN SPECIES TABLE + MOVE-TO-FOLDER ───────────────────────────────
    if not st.session_state.unknown_data.empty:
        unknown_count = len(st.session_state.unknown_data)
        st.subheader(f"🤔 Unknown Species — {unknown_count} file{'s' if unknown_count != 1 else ''}")
        st.dataframe(
            st.session_state.unknown_data,
            use_container_width=True,
            hide_index=True,
            height=400
        )

        if st.button("**Create New Folder for Unknown Sound Files**",
                     key="btn_unk_folder", use_container_width=True):
            st.session_state.show_unk_folder_form = not st.session_state.show_unk_folder_form
            st.session_state.unk_folder_result = None

        if st.session_state.show_unk_folder_form:
            with st.form("unk_folder_form"):
                st.markdown("**Enter the full path for the new unknown species folder**")
                st.markdown("The folder will be created if it does not exist, and files will be "
                            "**moved** from the source folder.")
                unk_folder_path = st.text_input(
                    "Destination folder path *",
                    placeholder=r"e.g.  /Users/you/Unknown_Bats  or  C:\Unknown_Bats"
                )
                col_save_unk, col_cancel_unk = st.columns(2)
                with col_save_unk:
                    confirm_unk = st.form_submit_button("**Create Folder & Move Files**", use_container_width=True)
                with col_cancel_unk:
                    cancel_unk = st.form_submit_button("**Cancel**", use_container_width=True)

                if cancel_unk:
                    st.session_state.show_unk_folder_form = False
                    st.rerun()

                if confirm_unk:
                    import os, shutil
                    raw_unk = (unk_folder_path or "").strip()
                    if raw_unk:
                        try:
                            unk_folder_path = str(Path(raw_unk).expanduser().resolve())
                        except Exception:
                            unk_folder_path = raw_unk
                    else:
                        unk_folder_path = ""

                    if not unk_folder_path:
                        st.error("Please enter a destination folder path.")
                    elif os.path.abspath(unk_folder_path) == os.path.abspath(st.session_state.source_folder):
                        st.error("Destination must be different from the source folder.")
                    else:
                        unknown_names = st.session_state.unknown_data['Filename'].tolist()
                        try:
                            os.makedirs(unk_folder_path, exist_ok=True)
                        except Exception as exc:
                            st.error(f"Could not create folder: {exc}")
                            st.stop()
                        moved, failed = [], []
                        for fname in unknown_names:
                            src = os.path.join(st.session_state.source_folder, fname)
                            dst = os.path.join(unk_folder_path, fname)
                            if not os.path.isfile(src):
                                failed.append(f"{fname} (not found in source)")
                                continue
                            try:
                                shutil.move(src, dst)
                                moved.append(fname)
                            except Exception as exc:
                                failed.append(f"{fname} ({exc})")
                        st.session_state.unk_folder_result = {
                            "path": unk_folder_path,
                            "moved": moved,
                            "failed": failed,
                        }
                        st.session_state.show_unk_folder_form = False
                        st.rerun()

        if st.session_state.unk_folder_result:
            res = st.session_state.unk_folder_result
            if res["moved"]:
                st.success(
                    f"Moved **{len(res['moved'])}** file(s) to `{res['path']}`."
                )
            if res["failed"]:
                st.warning("The following files could not be moved:\n" +
                           "\n".join(f"• {f}" for f in res["failed"]))

    st.markdown("---")

    # ── RESULTS CSV EXPORT ───────────────────────────────────────────────────
    if not st.session_state.export_data.empty:
        st.subheader("📄 Classification Results CSV")

        st.dataframe(
            st.session_state.export_data,
            use_container_width=True,
            hide_index=True,
            height=350
        )

        csv_bytes = convert_df_to_csv(st.session_state.export_data)
        st.download_button(
            label="Download Classification Results CSV",
            data=csv_bytes,
            file_name="batlab_classification_results.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_results_csv"
        )

    st.markdown("---")

# ============================================================================
# TAB 2: ADD DETECTOR
# ============================================================================
with tab2:
    st.markdown("---")
    st.header("Register a New Detector")
    st.markdown("---")

    with st.form("add_detector_form", clear_on_submit=False):
        name = st.text_input("Detector ID *", placeholder="e.g., Detector-A1")
        lat = st.text_input("Latitude *", placeholder="e.g., -33.9249")
        lon = st.text_input("Longitude *", placeholder="e.g., 18.4241")

        submitted_detector = st.form_submit_button("Save Detector", use_container_width=True)

        if submitted_detector:
            errors = []

            name = (name or "").strip()
            lat = (lat or "").strip()
            lon = (lon or "").strip()

            if not name:
                errors.append("Detector ID is required.")
            if not lat:
                errors.append("Latitude is required.")
            else:
                try:
                    lat_val = float(lat)
                    if not (-90 <= lat_val <= 90):
                        errors.append("Latitude must be between -90 and 90.")
                except ValueError:
                    errors.append("Latitude must be a valid number.")

            if not lon:
                errors.append("Longitude is required.")
            else:
                try:
                    lon_val = float(lon)
                    if not (-180 <= lon_val <= 180):
                        errors.append("Longitude must be between -180 and 180.")
                except ValueError:
                    errors.append("Longitude must be a valid number.")

            if not errors:
                # Check for duplicate: same ID, latitude, and longitude
                duplicate = any(
                    d["Detector ID"] == name and
                    float(d["Latitude"]) == float(lat) and
                    float(d["Longitude"]) == float(lon)
                    for d in st.session_state.detectors
                )
                if duplicate:
                    errors.append(
                        f"A detector with ID '{name}', Latitude '{lat}', and Longitude '{lon}' already exists."
                    )

            if errors:
                for e in errors:
                    st.error(e)
            else:
                # Update in-memory list
                st.session_state.detectors.append({
                    "Detector ID": name,
                    "Latitude": lat,
                    "Longitude": lon,
                })

                # Persist to database using existing MySQL helper
                db_detectors = [{
                    "Detector": name,
                    "Latitude": lat,
                    "Longitude": lon,
                }]
                db_errors = save_detectors(db_detectors)
                if db_errors:
                    for msg in db_errors:
                        st.error(f"Database error (detector): {msg}")
                else:
                    st.success("New detector saved to database successfully.")

                st.rerun()

    st.markdown("---")

    # Show registered detectors
    if st.session_state.detectors:
        st.subheader("Registered Detectors")
        detectors_df = pd.DataFrame(st.session_state.detectors)
        edited_detectors_df = st.data_editor(
            detectors_df,
            use_container_width=True,
            hide_index=True,
            height=400,
            disabled=["Detector ID"],
            num_rows="dynamic",
            key="detectors_editor",
        )
        if not edited_detectors_df.equals(detectors_df):
            before_rows = detectors_df.to_dict(orient="records")
            after_rows = edited_detectors_df.to_dict(orient="records")
            before_by_id = {
                str(r.get("Detector ID", "")).strip(): r
                for r in before_rows
                if str(r.get("Detector ID", "")).strip()
            }
            after_by_id = {
                str(r.get("Detector ID", "")).strip(): r
                for r in after_rows
                if str(r.get("Detector ID", "")).strip()
            }

            deleted_ids = [det_id for det_id in before_by_id if det_id not in after_by_id]
            added_rows = []
            for det_id, after in after_by_id.items():
                if det_id in before_by_id:
                    continue
                added_rows.append(
                    {
                        "Detector": det_id,
                        "Latitude": after.get("Latitude", ""),
                        "Longitude": after.get("Longitude", ""),
                    }
                )
            changed_rows = []
            for det_id, before in before_by_id.items():
                if det_id not in after_by_id:
                    continue
                after = after_by_id[det_id]
                if str(before.get("Latitude", "")) != str(after.get("Latitude", "")) or str(
                    before.get("Longitude", "")
                ) != str(after.get("Longitude", "")):
                    changed_rows.append(
                        {
                            "Detector": det_id,
                            "Latitude": after.get("Latitude", ""),
                            "Longitude": after.get("Longitude", ""),
                        }
                    )

            db_errors = []
            if deleted_ids:
                db_errors.extend(delete_detectors(deleted_ids))
            if added_rows:
                db_errors.extend(save_detectors(added_rows))
            if changed_rows:
                db_errors.extend(save_detectors(changed_rows))

            if db_errors:
                for msg in db_errors:
                    st.error(f"Database error (detector edit): {msg}")
            else:
                if deleted_ids:
                    st.success(f"Deleted {len(deleted_ids)} detector row(s).")
                if added_rows:
                    st.success(f"Added {len(added_rows)} detector row(s).")
                if changed_rows:
                    st.success(f"Saved {len(changed_rows)} detector edit(s).")
                st.session_state.detectors = edited_detectors_df.to_dict(orient="records")
                st.rerun()
    else:
        st.info("No detectors registered yet. Add your first detector above!")

# ============================================================================
# TAB 3: ADD SPECIES
# ============================================================================
with tab3:
    st.markdown("---")
    st.header("Register a New Species")
    st.markdown("---")

    # Load species from the database once for this tab
    db_species_for_tab, db_species_errors = load_species_from_db()
    if db_species_errors:
        for msg in db_species_errors:
            st.error(f"Database error (load species): {msg}")

    with st.form("add_species_form", clear_on_submit=False):
        abbr = st.text_input("Abbreviation *", placeholder="e.g., MYLU")
        latin = st.text_input("Latin Name *", placeholder="e.g., Myotis lucifugus")
        common = st.text_input("Common Name (optional)", placeholder="e.g., Little brown bat")

        submitted_species = st.form_submit_button("Save Species", use_container_width=True)

        if submitted_species:
            errors = []

            abbr = (abbr or "").strip()
            latin = (latin or "").strip()
            common = (common or "").strip()

            # ── Validation ───────────────────────────────────────────────────
            if not abbr:
                errors.append("Abbreviation is required.")
            else:
                if not abbr.isalnum():
                    errors.append("Abbreviation must contain only letters and numbers.")
                elif len(abbr) > 16:
                    errors.append("Abbreviation must be ≤ 16 characters.")

            if not latin:
                errors.append("Latin Name is required.")

            if not errors:
                # Check duplicates against what is persisted in the database,
                # not against any manifest-derived placeholders in session state.
                duplicate = any(
                    s["Abbreviation"].lower() == abbr.lower()
                    and s["Latin Name"].lower() == latin.lower()
                    for s in db_species_for_tab
                )
                if duplicate:
                    errors.append(
                        f"A species with abbreviation '{abbr}' and Latin name '{latin}' already exists."
                    )

            if errors:
                for e in errors:
                    st.error(e)
            else:
                # Update in-memory list
                st.session_state.species.append({
                    "Abbreviation": abbr,
                    "Latin Name": latin,
                    "Common Name": common,
                })

                # Persist to database using existing MySQL helper
                db_species = [{
                    "Abbreviation": abbr,
                    "LatinName": latin,
                    "CommonName": common,
                }]
                db_errors = save_species(db_species)
                if db_errors:
                    for msg in db_errors:
                        st.error(f"Database error (species): {msg}")
                else:
                    st.success("New species saved to database successfully.")

                st.rerun()

    st.markdown("---")

    # Show registered species (prefer database, not manifest placeholders)
    st.subheader("Registered Species")
    if db_species_for_tab:
        species_df = pd.DataFrame(db_species_for_tab)
    else:
        # Show an empty table with expected columns so new species appear
        species_df = pd.DataFrame(columns=["Abbreviation", "Latin Name", "Common Name"])

    edited_species_df = st.data_editor(
        species_df,
        use_container_width=True,
        hide_index=True,
        height=400,
        disabled=["Abbreviation"],
        num_rows="dynamic",
        key="species_editor",
    )
    if not edited_species_df.equals(species_df):
        before_rows = species_df.to_dict(orient="records")
        after_rows = edited_species_df.to_dict(orient="records")
        before_by_abbr = {
            str(r.get("Abbreviation", "")).strip(): r
            for r in before_rows
            if str(r.get("Abbreviation", "")).strip()
        }
        after_by_abbr = {
            str(r.get("Abbreviation", "")).strip(): r
            for r in after_rows
            if str(r.get("Abbreviation", "")).strip()
        }

        deleted_abbr = [abbr for abbr in before_by_abbr if abbr not in after_by_abbr]
        changed_rows = []
        for abbr, before in before_by_abbr.items():
            if abbr not in after_by_abbr:
                continue
            after = after_by_abbr[abbr]
            if (
                str(before.get("Latin Name", "")) != str(after.get("Latin Name", ""))
                or str(before.get("Common Name", "")) != str(after.get("Common Name", ""))
            ):
                changed_rows.append(
                    {
                        "Abbreviation": abbr,
                        "LatinName": after.get("Latin Name", ""),
                        "CommonName": after.get("Common Name", ""),
                    }
                )

        db_errors = []
        if deleted_abbr:
            db_errors.extend(delete_species(deleted_abbr))
        if changed_rows:
            db_errors.extend(save_species(changed_rows))

        if db_errors:
            for msg in db_errors:
                st.error(f"Database error (species edit): {msg}")
        else:
            if deleted_abbr:
                st.success(f"Deleted {len(deleted_abbr)} species row(s).")
            if changed_rows:
                st.success(f"Saved {len(changed_rows)} species edit(s).")
            st.session_state.species = edited_species_df.to_dict(orient="records")
            st.rerun()

# ============================================================================
# TAB 4: ADD TRAINING DATA
# ============================================================================
with tab4:
    st.markdown("---")
    st.header("Add Training Data")
    st.markdown("---")

    col_species, col_location = st.columns(2)

    with col_species:
        st.markdown("**Select Species:**")
        species_search = st.text_input(
            "Search species",
            placeholder="Type to filter...",
            key="species_search"
        )

        all_species = [s['Abbreviation'] for s in st.session_state.species]
        filtered_species = [sp for sp in all_species
                            if species_search.lower() in sp.lower()] if species_search else all_species

        selected_species = st.radio(
            "Species options",
            options=filtered_species if filtered_species else ["No species registered yet"],
            label_visibility="collapsed",
            key="species_radio"
        )

    with col_location:
        st.markdown("**Select Detector:**")
        detector_search = st.text_input(
            "Search detector",
            placeholder="Type to filter...",
            key="detector_search"
        )

        all_detectors = [d["Detector ID"] for d in st.session_state.detectors]
        filtered_detectors = [d for d in all_detectors
                              if detector_search.lower() in d.lower()] if detector_search else all_detectors

        selected_detector = st.radio(
            "Detector options",
            options=filtered_detectors if filtered_detectors else ["No detectors registered yet"],
            label_visibility="collapsed",
            key="detector_radio"
        )

    st.markdown("---")

    st.markdown("**Upload Training Audio Files (.wav, max 200MB per file):**")
    training_files = st.file_uploader(
        "Drop .wav files here",
        type=['wav'],
        accept_multiple_files=True,
        key=f'training_file_uploader_{st.session_state.training_uploader_key}',
        label_visibility="collapsed"
    )

    if training_files:
        oversized_files = []
        for f in training_files:
            file_size_mb = f.size / (1024 * 1024)
            if file_size_mb > 200:
                oversized_files.append(f"{f.name} ({file_size_mb:.1f}MB)")
        if oversized_files:
            st.error(f"The following files exceed 200MB limit: {', '.join(oversized_files)}")
        else:
            st.info(f"{len(training_files)} training file(s) selected")

    st.markdown("---")

    submitted_training = st.button("Save Training Data", use_container_width=True, type="primary", key="btn_save_training_data")

    if submitted_training:
        errors = []
        if not st.session_state.species:
            errors.append("No species registered. Please register at least one species first.")
        if not st.session_state.detectors:
            errors.append("No detectors registered. Please register at least one detector first.")
        if not training_files:
            errors.append("Please upload at least one .wav file.")
        else:
            oversized = [f for f in training_files if f.size / (1024 * 1024) > 200]
            if oversized:
                errors.append("Cannot save: Some files exceed the 200MB limit.")
        if errors:
            for e in errors:
                st.error(e)
        else:
            new_file_names = []
            for f in training_files:
                file_bytes = f.read()
                st.session_state.training_file_bytes[f.name] = file_bytes
                new_file_names.append(f.name)

            entry = {
                "Species": selected_species,
                "Detector": selected_detector,
                "FileCount": len(training_files),
                "FileNames": new_file_names,
            }

            # Persist to Call_Library (save_training_data)
            db_entries = [
                {
                    "Species": entry["Species"],
                    "Location": entry["Detector"],
                    "FileCount": entry["FileCount"],
                    "FileNames": entry["FileNames"],
                }
            ]
            bytes_map = {
                name: st.session_state.training_file_bytes.get(name)
                for name in new_file_names
            }
            db_errors = save_training_data(db_entries, bytes_map)

            for msg in db_errors:
                st.error(f"Database error (training data): {msg}")

            if not db_errors:
                st.success(
                    f"Training data saved to database: {len(training_files)} file(s) for {selected_species} at detector '{selected_detector}'."
                )

            st.session_state.training_uploader_key += 1
            st.rerun()

    st.markdown("---")

    # Show training data entries from database (aggregated)
    records_df, records_errors = load_training_records_df()
    if records_errors:
        for msg in records_errors:
            st.error(f"Database error (training records): {msg}")
    if not records_df.empty:
        st.subheader("Training Data Entries (from database)")
        summary_df = (
            records_df.groupby(["species_abbreviation", "location_name"], as_index=False)["file_count"]
            .sum()
            .rename(
                columns={
                    "species_abbreviation": "Species",
                    "location_name": "Detector",
                    "file_count": "Files",
                }
            )
        )
        edited_summary_df = st.data_editor(
            summary_df,
            use_container_width=True,
            hide_index=True,
            height=400,
            disabled=["Files"],
            num_rows="dynamic",
            key="training_data_editor",
        )
        if not edited_summary_df.equals(summary_df):
            before_rows = summary_df.to_dict(orient="records")
            after_rows = edited_summary_df.to_dict(orient="records")
            before_by_key = {
                (
                    str(r.get("Species", "")).strip(),
                    str(r.get("Detector", "")).strip(),
                ): r
                for r in before_rows
                if str(r.get("Species", "")).strip() and str(r.get("Detector", "")).strip()
            }
            after_keys = {
                (
                    str(r.get("Species", "")).strip(),
                    str(r.get("Detector", "")).strip(),
                )
                for r in after_rows
                if str(r.get("Species", "")).strip() and str(r.get("Detector", "")).strip()
            }

            updates = []
            for before, after in zip(before_rows, after_rows):
                old_species = str(before.get("Species", "")).strip()
                old_detector = str(before.get("Detector", "")).strip()
                new_species = str(after.get("Species", "")).strip()
                new_detector = str(after.get("Detector", "")).strip()
                file_count = before.get("Files", 0)
                if old_species != new_species or old_detector != new_detector:
                    updates.append(
                        {
                            "old_species": old_species,
                            "new_species": new_species,
                            "old_detector": old_detector,
                            "new_detector": new_detector,
                            "file_count": int(file_count),
                        }
                    )

            deletions = []
            for (species, detector), row in before_by_key.items():
                if (species, detector) not in after_keys:
                    deletions.append(
                        {
                            "species": species,
                            "detector": detector,
                            "file_count": int(row.get("Files", 0) or 0),
                        }
                    )

            db_errors = []
            if deletions:
                db_errors.extend(delete_call_library_data(deletions))
            if updates:
                db_errors.extend(update_call_library_data(updates))

            if db_errors:
                for msg in db_errors:
                    st.error(f"Database error (training data edit): {msg}")
            else:
                if deletions:
                    st.success(f"Deleted {len(deletions)} training-data row(s).")
                if updates:
                    st.success(f"Saved {len(updates)} training-data edit(s).")
                st.rerun()
    else:
        st.info("No training data entries found in the database yet.")

# ============================================================================
# TAB 5: TRAIN NEW MODEL
# ============================================================================
with tab5:
    st.markdown("---")
    st.header("Train New Model")
    st.markdown("---")

    # Precompute list of detector IDs (may be empty)
    all_detector_ids = [d["Detector ID"] for d in st.session_state.detectors]

    if not st.session_state.detectors:
        st.info("No detectors registered yet. Please add detectors in the 'Add Detector' tab first.")
    elif not st.session_state.species:
        st.info("No species registered yet. Please add species in the 'Add Species' tab first.")
    else:
        if "last_subset_training_job" not in st.session_state:
            st.session_state.last_subset_training_job = None
        if "last_subset_training_log" not in st.session_state:
            st.session_state.last_subset_training_log = None
        if "last_full_training_job" not in st.session_state:
            st.session_state.last_full_training_job = None
        if "last_full_training_log" not in st.session_state:
            st.session_state.last_full_training_log = None

        # ------------------------------------------------------------------
        # 1) Manage subset models
        # ------------------------------------------------------------------
        project_root = Path(__file__).resolve().parent
        local_models_dir = project_root / "model_checkpoints" / "local"
        local_models_dir.mkdir(parents=True, exist_ok=True)

        st.subheader("Manage subset models")
        subset_rows = []
        for subdir in sorted(local_models_dir.iterdir()):
            if not subdir.is_dir():
                continue
            pt = subdir / f"{subdir.name}.pt"
            if not pt.exists():
                continue
            meta = subdir / f"{subdir.name}_metadata.json"
            created_at = ""
            num_examples = ""
            if meta.exists():
                try:
                    import json

                    payload = json.loads(meta.read_text(encoding="utf-8") or "{}")
                    created_at = str(payload.get("created_at", "") or "")
                    num_examples = str(payload.get("num_examples", "") or "")
                except Exception:
                    pass
            subset_rows.append(
                {
                    "Model": subdir.name,
                    "Created": created_at,
                    "Examples": num_examples,
                    "Path": str(pt),
                }
            )

        subset_df = pd.DataFrame(subset_rows) if subset_rows else pd.DataFrame(
            columns=["Model", "Created", "Examples", "Path"]
        )
        edited_subset_df = st.data_editor(
            subset_df,
            use_container_width=True,
            hide_index=True,
            disabled=["Model", "Created", "Examples", "Path"],
            num_rows="dynamic",
            key="subset_models_editor",
        )

        if subset_rows and not edited_subset_df.equals(subset_df):
            before_models = [str(m).strip() for m in subset_df["Model"].tolist()]
            after_models = set(str(m).strip() for m in edited_subset_df.get("Model", []).tolist())
            deleted_models = [m for m in before_models if m and m not in after_models]

            if deleted_models:
                errors = []
                for model_name in deleted_models:
                    try:
                        shutil.rmtree(local_models_dir / model_name)
                    except Exception as exc:
                        errors.append(f"Failed to delete `{model_name}`: {exc}")

                if errors:
                    for msg in errors:
                        st.error(msg)
                else:
                    st.success(f"Deleted {len(deleted_models)} subset model(s).")
                    st.rerun()

        if not subset_rows:
            st.info("No subset models found under `model_checkpoints/local/` yet.")

        st.markdown("---")

        # ------------------------------------------------------------------
        # 2) Retrain full / large model
        # ------------------------------------------------------------------
        st.subheader("Retrain full model")
        st.markdown(
            "Use all available training data in the database to retrain the large base model."
        )

        full_model_name = st.text_input(
            "Optional full model name",
            placeholder="e.g. full_model_april",
            help="Leave blank to use a timestamped default name.",
            key="full_model_manual_name",
        )

        retrain_full_clicked = st.button(
            "Retrain Full Model",
            use_container_width=True,
            type="secondary",
            key="btn_retrain_full_model",
        )

        if st.session_state.last_full_training_job:
            job = st.session_state.last_full_training_job
            st.success(
                "Last full model training complete.\n\n"
                f"- Model name: `{job.get('model_name')}`\n"
                f"- Examples used: {job.get('num_examples')}`\n"
                f"- Full model saved to: `{job.get('output_model_path')}`\n"
                f"- Metadata: `{job.get('metadata_path')}`"
            )

        if retrain_full_clicked:
            full_log_buf = io.StringIO()

            class _FullStreamToCode:
                def __init__(self, buf: io.StringIO, placeholder, max_chars: int = 25000):
                    self.buf = buf
                    self.placeholder = placeholder
                    self.max_chars = max_chars
                    self._last_len = 0

                def write(self, s):
                    if not s:
                        return 0
                    self.buf.write(s)
                    cur = self.buf.getvalue()
                    if "\n" in s or (len(cur) - self._last_len) > 400:
                        tail = cur[-self.max_chars:]
                        self.placeholder.code(tail, language="text")
                        self._last_len = len(cur)
                    return len(s)

                def flush(self):
                    cur = self.buf.getvalue()
                    tail = cur[-self.max_chars:]
                    self.placeholder.code(tail, language="text")
                    self._last_len = len(cur)

            full_output_placeholder = None
            with st.expander("Full model training output", expanded=False):
                full_output_placeholder = st.empty()
                if st.session_state.last_full_training_log:
                    full_output_placeholder.code(
                        str(st.session_state.last_full_training_log)[-25000:],
                        language="text",
                    )

            full_stream = _FullStreamToCode(full_log_buf, full_output_placeholder) if full_output_placeholder else full_log_buf

            with st.spinner("Retraining full model. This may take several minutes..."):
                try:
                    project_root = Path(__file__).resolve().parent
                    call_df = get_call_library_data()

                    with contextlib.redirect_stdout(full_stream), contextlib.redirect_stderr(full_stream):
                        job = retrain_full_model_from_ui(
                            conn=None,
                            call_library_df=call_df,
                            model_name=full_model_name,
                        )

                    if hasattr(full_stream, "flush"):
                        full_stream.flush()

                    st.session_state.last_full_training_job = {
                        "model_name": job.model_name,
                        "num_examples": job.num_examples,
                        "output_model_path": job.output_model_path,
                        "metadata_path": job.metadata_path,
                    }
                    st.session_state.last_full_training_log = full_log_buf.getvalue() or ""

                    st.rerun()

                except Exception as e:
                    try:
                        if full_output_placeholder is not None:
                            full_output_placeholder.code(
                                (full_log_buf.getvalue() or "")[-25000:],
                                language="text",
                            )
                    except Exception:
                        pass
                    st.error(f"Full model training failed: {e}")

        st.markdown("---")

        # ------------------------------------------------------------------
        # 3) Choose subset training data (existing setup)
        # ------------------------------------------------------------------
        st.subheader("Choose subset training data")
        st.markdown("Select one or more detectors and the species you want to include in the new model training run.")
        st.markdown("---")

        # Load training metadata once to determine which species have training per detector.
        records_df, records_errors = load_training_records_df()
        if records_errors:
            for msg in records_errors:
                st.error(f"Database error (training records): {msg}")

        # Build a map of detector -> sorted list of species that have training data
        if records_df is not None and not records_df.empty:
            det_species_map: dict[str, list[str]] = (
                records_df.groupby("location_name")["species_abbreviation"]
                .apply(lambda s: sorted(set(s.dropna().astype(str))))
                .to_dict()
            )
        else:
            det_species_map = {}

        # Session state for multi-detector and multi-species selection
        if "train_selected_detectors" not in st.session_state:
            st.session_state.train_selected_detectors = []
        if "train_selected_species" not in st.session_state:
            st.session_state.train_selected_species = []

        # Detector search
        detector_search = st.text_input(
            "Search detectors",
            placeholder="Type to filter detectors...",
            key="train_detector_search"
        )
        st.markdown("---")

        filtered_detectors = [
            d for d in all_detector_ids
            if detector_search.lower() in d.lower()
        ] if detector_search else all_detector_ids

        if not filtered_detectors:
            st.info("No detectors match your search.")
            selected_detectors: list[str] = []
        else:
            # Display all (filtered) detectors from the database as checkboxes
            st.markdown("**Detectors found in database:**")
            selected_detectors = []
            for det_id in filtered_detectors:
                col_check, col_label = st.columns([0.05, 0.95])
                with col_check:
                    checked = st.checkbox(
                        "",
                        value=det_id in st.session_state.train_selected_detectors,
                        key=f"train_det_{det_id}",
                    )
                with col_label:
                    st.markdown(f"**{det_id}**")

                if checked:
                    selected_detectors.append(det_id)

            st.session_state.train_selected_detectors = selected_detectors

        st.markdown("---")

        # Species selection based on all selected detectors
        union_species: list[str] = []
        for det_id in selected_detectors:
            union_species.extend(det_species_map.get(det_id, []))
        union_species = sorted(set(union_species))

        if not selected_detectors:
            st.info("Select at least one detector above to see available species.")
        elif not union_species:
            st.warning("No training data in the database for the selected detectors yet.")
        else:
            st.markdown("**Species with training data for selected detector(s):**")
            selected_species: list[str] = []
            for sp in union_species:
                checked = st.checkbox(
                    sp,
                    value=sp in st.session_state.train_selected_species,
                    key=f"train_sp_{sp}",
                )
                if checked:
                    selected_species.append(sp)
            st.session_state.train_selected_species = selected_species

        st.markdown("---")

        manual_model_name = st.text_input(
            "Optional model name",
            placeholder="e.g. tcu_subset_march",
            help="Letters/numbers/underscore/dash only. If it already exists, a timestamp is appended.",
            key="subset_model_manual_name",
        )

        # Greyed out until at least one detector and one species are selected
        can_train = bool(
            st.session_state.train_selected_detectors
            and st.session_state.train_selected_species
        )

        subset_train_clicked = st.button(
            "Train Subset Model",
            use_container_width=True,
            type="primary",
            key="btn_train_model_global",
            disabled=not can_train,
        )

        # Show the last-run result near the Train button/output (not in Manage models).
        if st.session_state.last_subset_training_job:
            job = st.session_state.last_subset_training_job
            st.success(
                "Last subset model training complete.\n\n"
                f"- Model name: `{job.get('model_name')}`\n"
                f"- Examples used: {job.get('num_examples')}\n"
                f"- Subset model saved to: `{job.get('output_model_path')}`\n"
                f"- Metadata: `{job.get('metadata_path')}`"
            )

        if subset_train_clicked and can_train:
            log_buf = io.StringIO()

            class _StreamToCode:
                def __init__(self, buf: io.StringIO, placeholder, max_chars: int = 25000):
                    self.buf = buf
                    self.placeholder = placeholder
                    self.max_chars = max_chars
                    self._last_len = 0

                def write(self, s):
                    if not s:
                        return 0
                    self.buf.write(s)
                    # Update UI on newlines or sizable chunks
                    cur = self.buf.getvalue()
                    if "\n" in s or (len(cur) - self._last_len) > 400:
                        tail = cur[-self.max_chars :]
                        self.placeholder.code(tail, language="text")
                        self._last_len = len(cur)
                    return len(s)

                def flush(self):
                    cur = self.buf.getvalue()
                    tail = cur[-self.max_chars :]
                    self.placeholder.code(tail, language="text")
                    self._last_len = len(cur)

            output_placeholder = None
            # Always show (collapsed by default) after user clicks Train.
            with st.expander("Training output", expanded=False):
                output_placeholder = st.empty()
                # Seed with last run log if present
                if st.session_state.last_subset_training_log:
                    output_placeholder.code(
                        str(st.session_state.last_subset_training_log)[-25000:],
                        language="text",
                    )

            stream = _StreamToCode(log_buf, output_placeholder) if output_placeholder else log_buf

            with st.spinner("Training model. This may take several minutes..."):
                try:
                    # Build per-detector selection structure expected by subset trainer.
                    # Since the UI currently selects species globally, we apply the same
                    # chosen species list to each selected detector (intersected with
                    # what the DB says exists for that detector).
                    selected_species = list(st.session_state.train_selected_species or [])
                    train_selections: dict[str, dict] = {}
                    for det_id in st.session_state.train_selected_detectors:
                        available = set(det_species_map.get(det_id, []))
                        chosen_for_det = sorted({sp for sp in selected_species if sp in available})
                        train_selections[det_id] = {"selected": True, "species": chosen_for_det}

                    # Base model for fine-tuning:
                    #   1) newest full retrained model: model_checkpoints/full_retrained/<name>/<name>.pt
                    #   2) fallback colab base: model_checkpoints/colab/best_model.pt
                    # Subset model is saved to model_checkpoints/local/<subset_name>/<subset_name>.pt
                    project_root = Path(__file__).resolve().parent
                    cfg = Config.from_yaml(project_root / "configs" / "default.yaml")
                    model_base_dir = project_root / cfg.model_dir
                    colab_base_model = model_base_dir / "colab" / "best_model.pt"

                    # Prefer latest full_retrained checkpoint if available.
                    full_dir = model_base_dir / "full_retrained"
                    full_candidates: list[Path] = []
                    if full_dir.is_dir():
                        for subdir in full_dir.iterdir():
                            if not subdir.is_dir():
                                continue
                            pt = subdir / f"{subdir.name}.pt"
                            if pt.exists():
                                full_candidates.append(pt)

                    base_model_pt: Path | None = None
                    if full_candidates:
                        try:
                            base_model_pt = max(full_candidates, key=lambda p: p.stat().st_mtime)
                        except Exception:
                            base_model_pt = full_candidates[-1]
                    elif colab_base_model.exists():
                        base_model_pt = colab_base_model

                    if base_model_pt is None or not base_model_pt.exists():
                        st.error(
                            "No base model found for subset training. "
                            "Expected either:\n"
                            "- `model_checkpoints/full_retrained/<name>/<name>.pt` (preferred)\n"
                            "- `model_checkpoints/colab/best_model.pt` (fallback)"
                        )
                    else:
                        base_model_path = str(base_model_pt)
                        call_df = get_call_library_data()
                        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                            job = create_subset_model_from_ui_selection(
                                conn=None,
                                train_selections=train_selections,
                                base_model_path=base_model_path,
                                call_library_df=call_df,
                                model_name=manual_model_name,
                            )
                        if hasattr(stream, "flush"):
                            stream.flush()

                        st.session_state.last_subset_training_job = {
                            "model_name": job.model_name,
                            "num_examples": job.num_examples,
                            "output_model_path": job.output_model_path,
                            "metadata_path": job.metadata_path,
                        }
                        st.session_state.last_subset_training_log = log_buf.getvalue() or ""

                        # Rerun so the new model shows up in Manage subset models.
                        st.rerun()
                except Exception as e:
                    try:
                        if output_placeholder is not None:
                            output_placeholder.code(
                                (log_buf.getvalue() or "")[-25000:],
                                language="text",
                            )
                    except Exception:
                        pass
                    st.error(f"Training failed: {e}")
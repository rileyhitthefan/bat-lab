import streamlit as st
import pandas as pd
import time
from pathlib import Path

from src.ml.classify_app import classify_uploaded_files
from src.ml.config import Config
from src.db.connection import (
    get_call_library_data,
    save_detectors,
    save_species,
    save_training_data,
    load_detectors_from_db,
    load_species_from_db,
    load_training_records_df,
)
from src.ml.training.subset_model_trainer import create_subset_model_from_ui_selection

# ============================================================================
# MODEL DISCOVERY (for Classify tab dropdown)
# ============================================================================

def get_available_models() -> list[tuple[str, str]]:
    """
    Return list of (display_label, model_path_str) for the Classify tab:
    - Colab (base) model at model_checkpoints/colab/best_model.pt
    - All subset models under model_checkpoints/local/<name>/<name>.pt
    """
    project_root = Path(__file__).resolve().parent
    cfg = Config.from_yaml(project_root / "configs" / "default.yaml")
    base = project_root / cfg.model_dir
    options: list[tuple[str, str]] = []

    colab_pt = base / "colab" / "best_model.pt"
    if colab_pt.exists():
        options.append(("Colab (base model)", str(colab_pt)))

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


# Optional display names for species (manifest only has "label" = abbreviation)
SPECIES_DISPLAY_NAMES: dict[str, tuple[str, str]] = {
    "TAPMAU": ("Taphozous mauritianus", "Mauritian Tomb Bat"),
    "TADAEG": ("Tadarida aegyptiaca", "Egyptian Free-tailed Bat"),
    "OTOMAR": ("Otomops martiensseni", "Large-eared Free-tailed Bat"),
    "SCODIN": ("Scotophilus dinganii", "African Yellow Bat"),
    "MINNAT": ("Miniopterus natalensis", "Natal Long-fingered Bat"),
    "NEOCAP": ("Neoromicia capensis", "Cape Serotine Bat"),
    "MYOTRI": ("Myotis tricolor", "Temminck's Myotis"),
    "NYCTHE": ("Nycteris thebaica", "Egyptian Slit-faced Bat"),
    "RHICAP": ("Rhinolophus capensis", "Cape Horseshoe Bat"),
    "EPTHOT": ("Eptesicus hotentottus", "Long-tailed Serotine Bat"),
    "LAEBOT": ("Laephotis botswanae", "Botswana Long-eared Bat"),
    "SCOVIR": ("Scotophilus viridis", "Greenish Yellow Bat"),
}


def load_species_from_manifest(project_root: Path, config_path: Path | None = None) -> list[dict] | None:
    """
    Load species list from the data manifest (filepath, label, location).
    Tries project_root / config.manifest_csv, then project_root / scripts / data_manifest.csv.
    Returns list of {Abbreviation, Latin Name, Common Name}, or None if no manifest found/empty.
    """
    config_path = config_path or (project_root / "configs" / "default.yaml")
    if not config_path.exists():
        return None
    cfg = Config.from_yaml(config_path)
    
    manifest_path = project_root / cfg.manifest_csv
    if not manifest_path.exists():
        manifest_path = project_root / "scripts" / "data_manifest.csv"
    if not manifest_path.exists():
        return None
    try:
        df = pd.read_csv(manifest_path)
        if "label" not in df.columns or df.empty:
            return None
        labels = sorted(df["label"].dropna().astype(str).unique())
        if not labels:
            return None
        out = []
        for abbr in labels:
            latin, common = SPECIES_DISPLAY_NAMES.get(abbr, ("", ""))
            out.append({
                "Abbreviation": abbr,
                "Latin Name": latin or abbr,
                "Common Name": common or "",
            })
        return out
    except Exception:
        return None


# Inject base (light) styles
inject_css("styles.css")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'known_data' not in st.session_state:
    st.session_state.known_data = pd.DataFrame(columns=['Filename', 'Species Prediction', 'Confidence Level'])

if 'unknown_data' not in st.session_state:
    st.session_state.unknown_data = pd.DataFrame(columns=['Filename'])

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'training_entries' not in st.session_state:
    st.session_state.training_entries = []

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

if 'org_result_msg' not in st.session_state:
    st.session_state.org_result_msg = None

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


def process_audio_files(
    source_folder: str,
    wav_filenames: list[str],
    model_path: str | None = None,
):
    """
    Classify .wav files from the given folder using the ML model (classify_app).
    model_path: optional path to .pt checkpoint; if None, classify_app uses its default.
    Returns (known_df, unknown_df) with columns Filename, Species Prediction, Confidence Level.
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

    known_df, unknown_df = classify_uploaded_files(file_objects, model_path=model_path)

    if missing_or_skipped:
        unknown_df = pd.concat(
            [unknown_df, pd.DataFrame(missing_or_skipped)],
            ignore_index=True,
        )

    return known_df, unknown_df


def convert_df_to_csv(df):
    """Convert a DataFrame to CSV format for download."""
    return df.to_csv(index=False).encode('utf-8')


def organise_files(source_folder, identified_folder, unknown_folder,
                   known_filenames, unknown_filenames):
    """
    Move identified and unknown .wav files from source_folder into the
    respective destination folders.  Files are MOVED (not copied) so there
    is no duplication.

    Returns (moved_identified, moved_unknown, errors) as lists of strings.
    """
    import os, shutil

    moved_identified = []
    moved_unknown    = []
    errors           = []

    # Helper: create folder if it does not exist
    def ensure_dir(path):
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as exc:
            errors.append(f"Could not create folder '{path}': {exc}")
            return False

    if not os.path.isdir(source_folder):
        errors.append(f"Source folder not found: '{source_folder}'")
        return moved_identified, moved_unknown, errors

    if not ensure_dir(identified_folder):
        return moved_identified, moved_unknown, errors

    if not ensure_dir(unknown_folder):
        return moved_identified, moved_unknown, errors

    def move_files(filenames, dest_folder, moved_list):
        for fname in filenames:
            src = os.path.join(source_folder, fname)
            dst = os.path.join(dest_folder, fname)

            if not os.path.isfile(src):
                # File might already have been moved in a previous run
                if os.path.isfile(dst):
                    moved_list.append(f"{fname} (already in destination)")
                else:
                    errors.append(f"File not found in source: '{fname}'")
                continue

            if os.path.abspath(src) == os.path.abspath(dst):
                moved_list.append(f"{fname} (already in destination)")
                continue

            try:
                shutil.move(src, dst)
                moved_list.append(fname)
            except Exception as exc:
                errors.append(f"Could not move '{fname}': {exc}")

    move_files(known_filenames,   identified_folder, moved_identified)
    move_files(unknown_filenames, unknown_folder,    moved_unknown)

    return moved_identified, moved_unknown, errors


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
    if not st.session_state.known_data.empty or not st.session_state.unknown_data.empty:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("**Start New Session**", use_container_width=True, key="btn_start_new_session"):
                st.session_state.known_data = pd.DataFrame(columns=['Filename', 'Species Prediction', 'Confidence Level'])
                st.session_state.unknown_data = pd.DataFrame(columns=['Filename'])
                st.session_state.uploaded_files = []
                st.session_state.org_result_msg = None
                st.session_state.source_folder  = ""
                st.session_state.show_id_folder_form  = False
                st.session_state.show_unk_folder_form = False
                st.session_state.id_folder_result  = None
                st.session_state.unk_folder_result = None
                st.session_state.file_uploader_key += 1
                st.success("✅ Session reset! Ready for new files.")
                st.rerun()

    st.markdown("---")

    # ── STEP 1: Model Selector ─────────────────────────────────────────────────────
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
                known_df, unknown_df = process_audio_files(
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
        st.dataframe(
            pd.DataFrame(st.session_state.detectors),
            use_container_width=True,
            hide_index=True,
            height=400
        )
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

    st.dataframe(
        species_df,
        use_container_width=True,
        hide_index=True,
        height=400
    )

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
            st.session_state.training_entries.append(entry)

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
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            height=400,
        )
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

        # Universal Train Model button at the bottom; greyed out until at least
        # one detector and one species are selected
        can_train = bool(
            st.session_state.train_selected_detectors
            and st.session_state.train_selected_species
        )
        clicked = st.button(
            "Train Model",
            use_container_width=True,
            type="primary",
            key="btn_train_model_global",
            disabled=not can_train,
        )

        if clicked and can_train:
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

                    # Base model for fine-tuning: model_checkpoints/colab/best_model.pt
                    # Subset model is saved to model_checkpoints/local/<subset_name>.pt
                    project_root = Path(__file__).resolve().parent
                    cfg = Config.from_yaml(project_root / "configs" / "default.yaml")
                    base_model_path = str(project_root / cfg.model_dir / "colab" / "best_model.pt")
                    if not Path(base_model_path).exists():
                        st.error(
                            f"Base model not found at {base_model_path}. "
                            "Ensure model_checkpoints/colab/best_model.pt exists for subset training."
                        )
                    else:
                        call_df = get_call_library_data()
                        job = create_subset_model_from_ui_selection(
                            conn=None,
                            train_selections=train_selections,
                            base_model_path=base_model_path,
                            call_library_df=call_df,
                        )

                        st.success(
                            "Subset model training complete.\n\n"
                            f"- Model name: `{job.model_name}`\n"
                            f"- Examples used: {job.num_examples}\n"
                            f"- Subset model saved to: `{job.output_model_path}`\n"
                            f"- Metadata: `{job.metadata_path}`"
                        )
                except Exception as e:
                    st.error(f"Training failed: {e}")
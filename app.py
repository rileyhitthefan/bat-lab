import streamlit as st
import pandas as pd
import time

from src.ui import inject_styles
from src.ml.classify_app import classify_uploaded_files

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(page_title="Bat Acoustic Identification", layout="wide")

# Inject base styles (light mode / login page)
inject_styles(dark_theme=False)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'known_data' not in st.session_state:
    st.session_state.known_data = pd.DataFrame(columns=['Filename', 'Species Prediction', 'Confidence Level'])

if 'unknown_data' not in st.session_state:
    st.session_state.unknown_data = pd.DataFrame(columns=['Filename'])

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'uploaded_file_bytes' not in st.session_state:
    st.session_state.uploaded_file_bytes = {}  # filename -> bytes

if 'training_entries' not in st.session_state:
    st.session_state.training_entries = []

if 'training_file_bytes' not in st.session_state:
    st.session_state.training_file_bytes = {}  # filename -> bytes (accumulated across submissions)

if 'detectors' not in st.session_state:
    st.session_state.detectors = []

if 'species' not in st.session_state:
    st.session_state.species = [
        {"Abbreviation": "TAPMAU", "Latin Name": "", "Common Name": ""},
        {"Abbreviation": "TADAEG", "Latin Name": "", "Common Name": ""},
        {"Abbreviation": "OTOMAR", "Latin Name": "", "Common Name": ""},
        {"Abbreviation": "SCODIN", "Latin Name": "", "Common Name": ""},
        {"Abbreviation": "MINNAT", "Latin Name": "", "Common Name": ""},
        {"Abbreviation": "NEOCAP", "Latin Name": "", "Common Name": ""},
        {"Abbreviation": "MYOTRI", "Latin Name": "", "Common Name": ""},
        {"Abbreviation": "NYCTHE", "Latin Name": "", "Common Name": ""},
        {"Abbreviation": "RHICAP", "Latin Name": "", "Common Name": ""},
    ]

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

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

def process_audio_files(uploaded_files):
    """Process uploaded audio files and classify them as KNOWN or UNKNOWN using the ML model."""
    return classify_uploaded_files(uploaded_files)


def convert_df_to_csv(df):
    """Convert a DataFrame to CSV format for download."""
    return df.to_csv(index=False).encode('utf-8')


def build_zip(filenames, file_bytes_dict):
    """
    Build an in-memory zip containing the given filenames.
    Returns bytes of the zip, or None if no files could be found.
    """
    import io, zipfile
    buf = io.BytesIO()
    found = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in filenames:
            if fname in file_bytes_dict:
                zf.writestr(fname, file_bytes_dict[fname])
                found += 1
    buf.seek(0)
    return buf.read() if found > 0 else None


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
        st.image("batlablogo.PNG", use_container_width =True)
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
    inject_styles(dark_theme=True)

# Header
st.title("Welcome to the BatLab!")
st.markdown("Analyze bat acoustic calls and identify species using machine learning")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Classify", "Add Detector", "Add Species", "Add Training Data"])

# ============================================================================
# TAB 1: CLASSIFY
# ============================================================================
with tab1:
    st.markdown("---")
    st.header("Upload and Classify Bat Acoustic Calls")
    
    # Add Start New Session button only when there's data
    if not st.session_state.known_data.empty or not st.session_state.unknown_data.empty:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("**Start New Session**", use_container_width=True):
                st.session_state.known_data = pd.DataFrame(columns=['Filename', 'Species Prediction', 'Confidence Level'])
                st.session_state.unknown_data = pd.DataFrame(columns=['Filename'])
                st.session_state.uploaded_files = []
                st.session_state.uploaded_file_bytes = {}
                st.session_state.org_result_msg = None
                st.session_state.source_folder  = ""
                st.session_state.show_id_folder_form  = False
                st.session_state.show_unk_folder_form = False
                st.session_state.id_folder_result  = None
                st.session_state.unk_folder_result = None
                # Increment the key to force recreate the file uploader
                st.session_state.file_uploader_key += 1
                st.success("✅ Session reset! Ready for new files.")
                st.rerun()
    
    st.markdown("---")

    with st.form("file_upload_form"):
        uploaded = st.file_uploader(
            "Choose .wav files (max 200MB each)",
            type=['wav'],
            accept_multiple_files=True,
            key=f'main_file_uploader_{st.session_state.file_uploader_key}'
        )

        submitted = st.form_submit_button("Classify", use_container_width=True)

        if submitted:
            if not uploaded:
                st.warning("Please upload at least one file before classifying.")
            else:
                oversized = []
                for f in uploaded:
                    size_mb = f.size / (1024 * 1024)
                    if size_mb > 200:
                        oversized.append(f"{f.name} ({size_mb:.1f} MB)")

                if oversized:
                    st.error(f"The following files exceed the 200MB limit: {', '.join(oversized)}")
                else:
                    with st.spinner("Analyzing audio files... Please wait."):
                        known_df, unknown_df = process_audio_files(uploaded)

                        if not known_df.empty:
                            st.session_state.known_data = pd.concat(
                                [st.session_state.known_data, known_df], ignore_index=True
                            )
                        if not unknown_df.empty:
                            st.session_state.unknown_data = pd.concat(
                                [st.session_state.unknown_data, unknown_df], ignore_index=True
                            )

                        st.session_state.uploaded_files.extend([f.name for f in uploaded])
                        for f in uploaded:
                            f.seek(0)
                            st.session_state.uploaded_file_bytes[f.name] = f.read()

                    st.success(f"✅ Classification complete! Processed {len(uploaded)} file(s).")
                    st.rerun()

    st.markdown("---")

    # -----------------------------------------------------------------------
    # IDENTIFIED SPECIES TABLE + FOLDER BUTTON
    # -----------------------------------------------------------------------
    if not st.session_state.known_data.empty:
        st.subheader("✅ Identified Species")
        st.dataframe(
            st.session_state.known_data,
            use_container_width=True,
            hide_index=True,
            height=400
        )

        # Button to reveal the folder form
        if st.button("**Create New Folder for Identified Sound Files**",
                     key="btn_id_folder", use_container_width=True):
            st.session_state.show_id_folder_form = not st.session_state.show_id_folder_form
            st.session_state.id_folder_result = None

        if st.session_state.show_id_folder_form:
            with st.form("id_folder_form"):
                st.markdown("**Name your new folder**")
                id_folder_name = st.text_input(
                    "Folder name *",
                    placeholder="e.g.  Identified_Bats_June2025"
                )
                col_save_id, col_cancel_id = st.columns(2)
                with col_save_id:
                    confirm_id = st.form_submit_button("**Create & Download**", use_container_width=True)
                with col_cancel_id:
                    cancel_id = st.form_submit_button("**Cancel**", use_container_width=True)

                if cancel_id:
                    st.session_state.show_id_folder_form = False
                    st.rerun()

                if confirm_id:
                    id_folder_name = (id_folder_name or "").strip()
                    if not id_folder_name:
                        st.error("Please enter a folder name.")
                    else:
                        known_names = st.session_state.known_data['Filename'].tolist()
                        zip_bytes = build_zip(
                            known_names,
                            st.session_state.uploaded_file_bytes
                        )
                        st.session_state.id_folder_result = {
                            "zip":        zip_bytes,
                            "folder_name": id_folder_name,
                            "count":      len(known_names),
                        }
                        st.session_state.show_id_folder_form = False
                        st.rerun()

        # Show download button once zip is ready
        if st.session_state.id_folder_result:
            res = st.session_state.id_folder_result
            if res["zip"]:
                st.success(
                    f"Folder **{res['folder_name']}** is ready — "
                    f"{res['count']} file(s) packed."
                )
                st.download_button(
                    label=f"**Download  {res['folder_name']}.zip**",
                    data=res["zip"],
                    file_name=f"{res['folder_name']}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="dl_id_zip"
                )
            else:
                st.warning(
                    "No audio data found to package. "
                    "Files may have been uploaded in a previous session — "
                    "please re-upload and classify them to use this feature."
                )

    else:
        st.info("No identified species yet. Upload files to get started!")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # UNKNOWN SPECIES TABLE + FOLDER BUTTON
    # -----------------------------------------------------------------------
    if not st.session_state.unknown_data.empty:
        st.subheader("❓ Unknown Species")
        st.dataframe(
            st.session_state.unknown_data,
            use_container_width=True,
            hide_index=True,
            height=400
        )

        # Button to reveal the folder form
        if st.button("**Create New Folder for Unknown Sound Files**",
                     key="btn_unk_folder", use_container_width=True):
            st.session_state.show_unk_folder_form = not st.session_state.show_unk_folder_form
            st.session_state.unk_folder_result = None

        if st.session_state.show_unk_folder_form:
            with st.form("unk_folder_form"):
                st.markdown("**Name your new folder**")
                unk_folder_name = st.text_input(
                    "Folder name *",
                    placeholder="e.g.  Unknown_Bats_June2025"
                )
                col_save_unk, col_cancel_unk = st.columns(2)
                with col_save_unk:
                    confirm_unk = st.form_submit_button("**Create & Download**", use_container_width=True)
                with col_cancel_unk:
                    cancel_unk = st.form_submit_button("**Cancel**", use_container_width=True)

                if cancel_unk:
                    st.session_state.show_unk_folder_form = False
                    st.rerun()

                if confirm_unk:
                    unk_folder_name = (unk_folder_name or "").strip()
                    if not unk_folder_name:
                        st.error("Please enter a folder name.")
                    else:
                        unknown_names = st.session_state.unknown_data['Filename'].tolist()
                        zip_bytes = build_zip(
                            unknown_names,
                            st.session_state.uploaded_file_bytes
                        )
                        st.session_state.unk_folder_result = {
                            "zip":         zip_bytes,
                            "folder_name": unk_folder_name,
                            "count":       len(unknown_names),
                        }
                        st.session_state.show_unk_folder_form = False
                        st.rerun()

        # Show download button once zip is ready
        if st.session_state.unk_folder_result:
            res = st.session_state.unk_folder_result
            if res["zip"]:
                st.success(
                    f"Folder **{res['folder_name']}** is ready — "
                    f"{res['count']} file(s) packed."
                )
                st.download_button(
                    label=f"**Download  {res['folder_name']}.zip**",
                    data=res["zip"],
                    file_name=f"{res['folder_name']}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="dl_unk_zip"
                )
            else:
                st.warning(
                    "No audio data found to package. "
                    "Files may have been uploaded in a previous session — "
                    "please re-upload and classify them to use this feature."
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

            if errors:
                for e in errors:
                    st.error(f"Error: {e}")
            else:
                st.session_state.detectors.append({
                    "Detector ID": name,
                    "Latitude": lat,
                    "Longitude": lon,
                })
                st.success("New detector saved successfully!")
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

            if errors:
                for e in errors:
                    st.error(f"Error: {e}")
            else:
                st.session_state.species.append({
                    "Abbreviation": abbr,
                    "Latin Name": latin,
                    "Common Name": common,
                })
                st.success("New species saved successfully!")
                st.rerun()

    st.markdown("---")

    # Show registered species
    if st.session_state.species:
        st.subheader("Registered Species")
        species_df = pd.DataFrame(st.session_state.species)
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

    placeholder_locations = ["Addo Elephant National Park", "Great Fish River Nature Reserve",
                             "Amakhala Game Reserve", "Tanglewood Conservation Area"]

    with st.form("add_training_form", clear_on_submit=False):
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
                options=filtered_species,
                label_visibility="collapsed",
                key="species_radio"
            )

        with col_location:
            st.markdown("**Select Location:**")
            location_search = st.text_input(
                "Search location",
                placeholder="Type to filter...",
                key="location_search"
            )

            filtered_locations = [loc for loc in placeholder_locations
                                  if
                                  location_search.lower() in loc.lower()] if location_search else placeholder_locations

            selected_location = st.radio(
                "Location options",
                options=filtered_locations,
                label_visibility="collapsed",
                key="location_radio"
            )

        st.markdown("---")

        st.markdown("**Upload Training Audio Files (.wav, max 200MB per file):**")
        training_files = st.file_uploader(
            "Drop .wav files here",
            type=['wav'],
            accept_multiple_files=True,
            key='training_file_uploader',
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

        col_save, col_cancel = st.columns(2)
        with col_save:
            submitted_training = st.form_submit_button("Save Training Data", use_container_width=True)
        with col_cancel:
            cancel_training = st.form_submit_button("Cancel", use_container_width=True)

        if submitted_training:
            if not training_files:
                st.error("Please upload at least one .wav file.")
            else:
                oversized = [f for f in training_files if f.size / (1024 * 1024) > 200]
                if oversized:
                    st.error("Cannot save: Some files exceed the 200MB limit.")
                else:
                    # Append file bytes into session state (accumulates across submissions)
                    new_file_names = []
                    for f in training_files:
                        file_bytes = f.read()
                        st.session_state.training_file_bytes[f.name] = file_bytes
                        new_file_names.append(f.name)

                    st.session_state.training_entries.append({
                        "Species": selected_species,
                        "Location": selected_location,
                        "FileCount": len(training_files),
                        "FileNames": new_file_names
                    })
                    total_files = len(st.session_state.training_file_bytes)
                    st.success(
                        f"Training data saved: {len(training_files)} file(s) for {selected_species} at {selected_location}. "
                        f"Total accumulated files: {total_files}")
                    st.rerun()

        if cancel_training:
            st.info("Cancelled - no changes made.")

    st.markdown("---")

    # Show training data entries
    if st.session_state.training_entries:
        st.subheader("Training Data Entries")
        total_accumulated = len(st.session_state.training_file_bytes)
        st.info(f"Total accumulated files in session: **{total_accumulated}**")
        training_df = pd.DataFrame([{
            "Species": entry["Species"],
            "Location": entry["Location"],
            "Files": entry["FileCount"]
        } for entry in st.session_state.training_entries])
        st.dataframe(
            training_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
    else:
        st.info("No training data entries yet. Add your first training dataset above!")
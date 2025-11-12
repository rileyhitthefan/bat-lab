import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from scipy.io import wavfile
import librosa
import time
import torch
import torch.nn.functional as F

MODEL_PATH = "src/classifier/model_it_1-2.pt"
CONFIG = {
    "sr": 48000,
    "mono": True,
    "duration_s": 1.0,
    "n_fft": 1024,
    "hop_length": 512,
    "n_mels": 128,
    "fmin": 1000,
    "fmax": 24000,
}

st.set_page_config(page_title="Bat Lab", layout="wide")

# Hide only the "Show/hide columns" button in st.dataframe/st.data_editor toolbars
st.markdown("""
<style>
/* Most builds */
button[title="Show/hide columns"],
button[aria-label="Show/hide columns"] { display: none !important; }

/* Scoped to dataframe/data_editor toolbars (newer testids) */
div[data-testid="stDataFrame"] button[title="Show/hide columns"],
div[data-testid="stDataFrame"] button[aria-label="Show/hide columns"],
div[data-testid="stDataEditor"] button[title="Show/hide columns"],
div[data-testid="stDataEditor"] button[aria-label="Show/hide columns"] { display: none !important; }

/* Fallbacks for different toolbar wrappers */
div[data-testid="stElementToolbar"] button[title*="Show/hide"],
div[data-testid="stElementToolbar"] button[aria-label*="Show/hide"] { display: none !important; }

/* If the popover/dialog is already open, hide it too */
div[role="dialog"][aria-label="Show/hide columns"],
div[role="menu"][aria-label="Show/hide columns"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# Hide only the "Download as CSV" button in st.dataframe toolbars
st.markdown("""
<style>
/* Most builds */
button[title="Download as CSV"],
button[aria-label="Download as CSV"] { display:none !important; }

/* Scoped to dataframe toolbars (newer testids) */
div[data-testid="stDataFrame"] button[title="Download as CSV"],
div[data-testid="stDataFrame"] button[aria-label="Download as CSV"] { display:none !important; }

/* Fallbacks for different toolbar wrappers */
div[data-testid="stElementToolbar"] button[title*="Download"],
div[data-testid="stElementToolbar"] button[aria-label*="Download"] { display:none !important; }
</style>
""", unsafe_allow_html=True)

if 'known_data' not in st.session_state:
    st.session_state.known_data = pd.DataFrame(columns=['FileName', 'SpeciesPrediction', 'ConfidenceLevel'])
if 'unknown_data' not in st.session_state:
    st.session_state.unknown_data = pd.DataFrame(columns=['FileName'])
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'training_entries' not in st.session_state:
    st.session_state.training_entries = []  # list of dicts
if 'show_add_training' not in st.session_state:
    st.session_state.show_add_training = False

@st.cache_resource
def cache_wav_file(file_name: str, file_bytes: bytes):
    buffer = BytesIO(file_bytes)
    sampling_rate, audio_data = wavfile.read(buffer)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    return file_name, sampling_rate, audio_data

@st.cache_resource
def load_model():
    import os
    from src.classifier.SmallAudioCNN import SmallAudioCNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    last_modified = os.path.getmtime(MODEL_PATH)
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    meta = checkpoint.get("meta", {})
    num_species = len(meta.get("species", [])) or 1
    num_locations = len(meta.get("locations", [])) or 2

    model = SmallAudioCNN(num_species, num_locations).to(device)
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    return model, meta, device, last_modified

@st.cache_data
def process_audio_files(cached_files):
    """
    Process uploaded audio files (cached tuples) and classify them using the trained PyTorch model.
    Each file is assigned a loc_id to match model input requirements.

    Args:
        cached_files: list of tuples (file_name, sampling_rate, audio_data)

    Returns:
        known_df, unknown_df: pandas DataFrames with classification results
    """
    model, meta, device, _ = load_model()
    known_results = []
    unknown_results = []

    # Determine number of locations from model
    num_locations = model.loc_embed.num_embeddings if hasattr(model, "loc_embed") else 1

    for i, (file_name, sr_orig, y_orig) in enumerate(cached_files):
        y = y_orig.astype(np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=1)  # convert to mono
        target_len = int(CONFIG["sr"] * CONFIG["duration_s"])
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mel = librosa.feature.melspectrogram(
            y=y, sr=CONFIG["sr"], n_fft=CONFIG["n_fft"], hop_length=CONFIG["hop_length"],
            n_mels=CONFIG["n_mels"], fmin=CONFIG["fmin"], fmax=CONFIG["fmax"]
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

        x = torch.tensor(mel_norm).unsqueeze(0).unsqueeze(0).to(device).float()
        # Assign a loc_id: cycle through available embeddings if multiple
        loc_id = torch.tensor([i % num_locations], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(x, loc_id)
            probs = F.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            confidence = conf.item()
            predicted_species = meta["species"][pred_idx.item()]
            print(meta)
            print(predicted_species, confidence)

        confidence_threshold = 0.5
        if confidence >= confidence_threshold:
            known_results.append({
                "FileName": file_name,
                "SpeciesPrediction": predicted_species,
                "ConfidenceLevel": f"{confidence * 100:.2f}%",
            })
        else:
            unknown_results.append({"FileName": file_name})

    known_df = pd.DataFrame(known_results)
    unknown_df = pd.DataFrame(unknown_results)
    return known_df, unknown_df

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

st.title("🦇 Bat Acoustic Identification System")
st.markdown("---")

st.subheader("Upload Audio Files")
uploaded_files = st.file_uploader(
    "Upload",
    type=['wav'],
    accept_multiple_files=True,
    key='file_uploader'
)
if uploaded_files:
    # remove cached files that are no longer uploaded
    st.session_state["uploaded_files"] = [
        f for f in st.session_state["uploaded_files"]
        if f[0] in [file.name for file in uploaded_files]
    ]
    for file in uploaded_files:
        if file.name not in [f[0] for f in st.session_state["uploaded_files"]]:
            file_name, sampling_rate, audio_data = cache_wav_file(file.name, file.read())
            st.session_state["uploaded_files"].append((file_name, sampling_rate, audio_data))
    st.info(f"📁 {len(uploaded_files)} file(s) uploaded")
    uploaded_files = st.session_state["uploaded_files"]

st.markdown("---")

if 'detectors' not in st.session_state:
    st.session_state.detectors = []
if 'show_add_detector' not in st.session_state:
    st.session_state.show_add_detector = False
if 'species' not in st.session_state:
    st.session_state.species = [
        {"Abbreviation": "TAPMAU", "LatinName": "", "CommonName": ""},
        {"Abbreviation": "TADAEG", "LatinName": "", "CommonName": ""},
        {"Abbreviation": "OTOMAR", "LatinName": "", "CommonName": ""},
        {"Abbreviation": "SCODIN", "LatinName": "", "CommonName": ""},
        {"Abbreviation": "MINNAT", "LatinName": "", "CommonName": ""},
        {"Abbreviation": "NEOCAP", "LatinName": "", "CommonName": ""},
        {"Abbreviation": "MYOTRI", "LatinName": "", "CommonName": ""},
        {"Abbreviation": "NYCTHE", "LatinName": "", "CommonName": ""},
        {"Abbreviation": "RHICAP", "LatinName": "", "CommonName": ""},
    ]
if 'show_add_species' not in st.session_state:
    st.session_state.show_add_species = False

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    # Classify button - triggers the ML model processing
    classify_button = st.button("🔍 Classify", type="primary", use_container_width=True)

with col2:
    # Add new sound detector button - now opens a small form
    add_detector_button = st.button("➕ Add New Sound Detector", use_container_width=True)

with col3:
    # Same look/feel as detector button
    add_species_button = st.button("➕ Add New Species", use_container_width=True)

with col4:
    # Add training data button
    add_training_button = st.button("➕ Add Training Data", use_container_width=True)

if classify_button:
    if uploaded_files:
        with st.spinner('Classifying audio files...'):
            time.sleep(1)
            known_df, unknown_df = process_audio_files(uploaded_files)
            st.session_state.known_data = known_df
            st.session_state.unknown_data = unknown_df
        st.success("✅ Classification complete!")
    else:
        st.warning("⚠️ Please upload audio files first.")

# Handle add detector button click -> toggle the form
if add_detector_button:
    st.session_state.show_add_detector = True

# Handle add species button click -> toggle the form
if add_species_button:
    st.session_state.show_add_species = True

# Handle add training data button click -> toggle the form
if add_training_button:
    st.session_state.show_add_training = True

# Detector input form (shown when toggled)
if st.session_state.show_add_detector:
    st.markdown("### ➕ Register a New Sound Detector")
    with st.form("add_detector_form", clear_on_submit=False):
        name = st.text_input("Name/ID *", placeholder="e.g., DET-Alpha-01")

        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat_str = st.text_input("Latitude (−90 to 90) *", placeholder="e.g., 40.4237")
        with col_lon:
            lon_str = st.text_input("Longitude (−180 to 180) *", placeholder="e.g., -86.9212")

        submitted = st.form_submit_button("💾 Save Detector", use_container_width=True)
        if submitted:
            errors = []

            # Validate required fields
            name = (name or "").strip()
            if not name:
                errors.append("Name is required.")

            try:
                lat = float((lat_str or "").strip())
                if not (-90.0 <= lat <= 90.0):
                    errors.append("Latitude must be between −90 and 90.")
            except Exception:
                errors.append("Latitude must be a valid number.")

            try:
                lon = float((lon_str or "").strip())
                if not (-180.0 <= lon <= 180.0):
                    errors.append("Longitude must be between −180 and 180.")
            except Exception:
                errors.append("Longitude must be a valid number.")

            if errors:
                for e in errors:
                    st.error(f"❌ {e}")
            else:
                st.session_state.detectors.append({
                    "Detector": name,
                    "Latitude": lat,
                    "Longitude": lon,
                })
                st.session_state.show_add_detector = False
                st.success("✅ New detector saved.")

# Species input form (shown when toggled)
if st.session_state.show_add_species:
    st.markdown("### ➕ Register a New Species")
    with st.form("add_species_form", clear_on_submit=False):
        abbr = st.text_input("Abbreviation *", placeholder="e.g., MYLU")
        latin = st.text_input("Latin Name (optional)", placeholder="e.g., Myotis lucifugus")
        common = st.text_input("Common Name (optional)", placeholder="e.g., Little brown bat")

        submitted_species = st.form_submit_button("💾 Save Species", use_container_width=True)
        if submitted_species:
            errors = []

            # ----- Required field + basic type/format checks -----
            abbr = (abbr or "").strip()
            latin = (latin or "").strip()
            common = (common or "").strip()

            if not abbr:
                errors.append("Abbreviation is required.")
            else:
                # Only accept letters and numbers
                if not abbr.isalnum():
                    errors.append("Abbreviation must contain only letters and numbers.")
                elif len(abbr) > 16:
                    errors.append("Abbreviation must be ≤ 16 characters.")

            # Latin Name and Common Name are optional, no validation needed
            # They're already strings from text_input

            # -----------------------------------------------------

            if errors:
                for e in errors:
                    st.error(f"❌ {e}")
            else:
                st.session_state.species.append({
                    "Abbreviation": abbr,
                    "LatinName": latin,
                    "CommonName": common,
                })
                st.session_state.show_add_species = False
                st.success("✅ New species saved.")

# Training data input form (shown when toggled) - NEW IMPLEMENTATION
if st.session_state.show_add_training:
    st.markdown("### ➕ Add Training Data")

    # Pre-populated locations (placeholders)
    placeholder_locations = ["Addo Elephant National Park", "Great Fish River Nature Reserve", "Amakhala Game Reserve", "Tanglewood Conservation Area"]

    with st.form("add_training_form", clear_on_submit=False):
        # Two columns for species and location selection
        col_species, col_location = st.columns(2)

        with col_species:
            st.markdown("**Select Species:**")
            # Search box for species
            species_search = st.text_input(
                "Search species",
                placeholder="Type to filter...",
                key="species_search"
            )

            # Filter species based on search - show only abbreviation
            all_species = [s['Abbreviation'] for s in st.session_state.species]
            filtered_species = [sp for sp in all_species
                                if species_search.lower() in sp.lower()] if species_search else all_species

            # Radio buttons for single selection (scrollable)
            selected_species = st.radio(
                "Species options",
                options=filtered_species,
                label_visibility="collapsed",
                key="species_radio"
            )

        with col_location:
            st.markdown("**Select Location:**")
            # Search box for locations
            location_search = st.text_input(
                "Search location",
                placeholder="Type to filter...",
                key="location_search"
            )

            # Filter locations based on search - use placeholder locations
            filtered_locations = [loc for loc in placeholder_locations
                                  if
                                  location_search.lower() in loc.lower()] if location_search else placeholder_locations

            # Radio buttons for single selection (scrollable)
            selected_location = st.radio(
                "Location options",
                options=filtered_locations,
                label_visibility="collapsed",
                key="location_radio"
            )

        st.markdown("---")

        # File uploader for training audio files
        st.markdown("**Upload Training Audio Files (.wav, max 200MB per file):**")
        training_files = st.file_uploader(
            "Drop .wav files here",
            type=['wav'],
            accept_multiple_files=True,
            key='training_file_uploader',
            label_visibility="collapsed"
        )

        if training_files:
            # Check file sizes
            oversized_files = []
            for f in training_files:
                file_size_mb = f.size / (1024 * 1024)  # Convert to MB
                if file_size_mb > 200:
                    oversized_files.append(f"{f.name} ({file_size_mb:.1f}MB)")

            if oversized_files:
                st.error(f"❌ The following files exceed 200MB limit: {', '.join(oversized_files)}")
            else:
                st.info(f"📁 {len(training_files)} training file(s) selected")

        st.markdown("---")

        # Save and Cancel buttons
        col_save, col_cancel = st.columns(2)
        with col_save:
            submitted_training = st.form_submit_button("💾 Save Training Data", use_container_width=True)
        with col_cancel:
            cancel_training = st.form_submit_button("❌ Cancel", use_container_width=True)

        if submitted_training:
            if not training_files:
                st.error("❌ Please upload at least one .wav file.")
            else:
                # Check for oversized files again before saving
                oversized = [f for f in training_files if f.size / (1024 * 1024) > 200]
                if oversized:
                    st.error("❌ Cannot save: Some files exceed the 200MB limit.")
                else:
                    # Store training entry
                    st.session_state.training_entries.append({
                        "Species": selected_species,
                        "Location": selected_location,
                        "FileCount": len(training_files),
                        "FileNames": [f.name for f in training_files]
                    })

                    st.session_state.show_add_training = False
                    st.success(
                        f"✅ Training data saved: {len(training_files)} file(s) for {selected_species} at {selected_location}")

        if cancel_training:
            st.session_state.show_add_training = False

# Show a compact summary of registered detectors (if any)
if st.session_state.detectors:
    st.caption("**Registered Detectors**")
    st.dataframe(
        pd.DataFrame(st.session_state.detectors),
        use_container_width=True,
        hide_index=True,
        height=180
    )

# Show a compact summary of training data entries (if any)
if st.session_state.training_entries:
    st.caption("**Training Data Entries**")
    training_df = pd.DataFrame([{
        "Species": entry["Species"],
        "Location": entry["Location"],
        "Files": entry["FileCount"]
    } for entry in st.session_state.training_entries])
    st.dataframe(
        training_df,
        use_container_width=True,
        hide_index=True,
        height=180
    )

st.markdown("---")

# ============================================================================
# RESULTS DISPLAY SECTION
# ============================================================================
st.subheader("Classification Results")

# Create two columns for side-by-side display of KNOWN and UNKNOWN results
col_known, col_unknown = st.columns(2)

with col_known:
    st.markdown("### 🟢 KNOWN")

    # Display KNOWN results in a scrollable dataframe
    if not st.session_state.known_data.empty:
        # Use st.dataframe with height parameter for scrollable window
        st.dataframe(
            st.session_state.known_data,
            use_container_width=True,
            height=400,
            hide_index=True
        )

        # Show count of known species
        st.caption(f"Total: {len(st.session_state.known_data)} identified species")
    else:
        st.info("No classified species yet. Upload and classify files to see results.")

with col_unknown:
    st.markdown("### 🔴 UNKNOWN")

    # Display UNKNOWN results in a scrollable dataframe
    if not st.session_state.unknown_data.empty:
        # Use st.dataframe with height parameter for scrollable window
        st.dataframe(
            st.session_state.unknown_data,
            use_container_width=True,
            height=400,
            hide_index=True
        )

        # Show count of unknown files
        st.caption(f"Total: {len(st.session_state.unknown_data)} unidentified files")
    else:
        st.info("No unclassified files.")

st.markdown("---")

# ============================================================================
# DOWNLOAD SECTION
# ============================================================================
st.subheader("Download Results")

col_download1, col_download2, col_download3 = st.columns([1, 1, 2])

with col_download1:
    # Download button for KNOWN results
    if not st.session_state.known_data.empty:
        csv_known = convert_df_to_csv(st.session_state.known_data)
        st.download_button(
            label="📥 Download Results (KNOWN)",
            data=csv_known,
            file_name="bat_identification_known.csv",
            mime="text/csv",
            use_container_width=True
        )

with col_download2:
    # Download button for UNKNOWN results
    if not st.session_state.unknown_data.empty:
        csv_unknown = convert_df_to_csv(st.session_state.unknown_data)
        st.download_button(
            label="📥 Download Results (UNKNOWN)",
            data=csv_unknown,
            file_name="bat_identification_unknown.csv",
            mime="text/csv",
            use_container_width=True
        )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <small>Bat Acoustic Identification System | Powered by Machine Learning</small>
</div>
""", unsafe_allow_html=True)
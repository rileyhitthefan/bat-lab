import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from scipy.io import wavfile
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
# Set the page layout to wide mode for better space utilization
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

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
# Initialize session state variables to persist data across reruns
# These variables store the classification results throughout the session
if 'known_data' not in st.session_state:
    st.session_state.known_data = pd.DataFrame(columns=['FileName', 'SpeciesPrediction', 'ConfidenceLevel'])

if 'unknown_data' not in st.session_state:
    st.session_state.unknown_data = pd.DataFrame(columns=['FileName'])

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
@st.cache_resource
def cache_wav_file(file_name: str, file_bytes: bytes):
    """
    Cache WAV file in memory for temporary processing.
    
    Args:
        file_name: Name of uploaded file.
        file_bytes: File contents.
    Returns:
        (file_name, sample_rate, audio_data)
    """
    buffer = BytesIO(file_bytes)
    sampling_rate, audio_data = wavfile.read(buffer)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    return file_name, sampling_rate, audio_data

@st.cache_data
def process_audio_files(uploaded_files):
    """
    Process uploaded audio files and classify them as KNOWN or UNKNOWN.

    This is a placeholder function where you would integrate your actual
    machine learning model for bat species identification.

    Args:
        uploaded_files: List of uploaded audio files from Streamlit

    Returns:
        tuple: (known_df, unknown_df) DataFrames with classification results
    """
    known_results = []
    unknown_results = []

    # Iterate through each uploaded file
    for file_data in uploaded_files:
        file_name, sampling_rate, audio_data = file_data

        # ====================================================================
        # MODEL INTEGRATION POINT
        # ====================================================================
        # TODO: Replace this section with your actual ML model prediction
        # Example integration:
        # 1. Load the audio file: audio_data = load_audio(file)
        # 2. Extract features: features = extract_features(audio_data)
        # 3. Make prediction: species, confidence = model.predict(features)
        # 4. Set threshold: if confidence > threshold, add to known, else unknown

        # Placeholder logic for demonstration
        # This simulates model predictions with dummy data
        import random

        # Simulate confidence score (0-100%)
        confidence = random.uniform(0.5, 0.99)

        # Define a confidence threshold (adjust based on your model)
        confidence_threshold = 0.75

        if confidence >= confidence_threshold:
            # High confidence - add to KNOWN category
            species_list = ['Myotis lucifugus', 'Eptesicus fuscus', 'Lasiurus borealis',
                            'Myotis septentrionalis', 'Perimyotis subflavus']
            predicted_species = random.choice(species_list)

            known_results.append({
                'FileName': file_name,
                'SpeciesPrediction': predicted_species,
                'ConfidenceLevel': f"{confidence * 100:.2f}%"
            })
        else:
            # Low confidence - add to UNKNOWN category
            unknown_results.append({
                'FileName': file_name
            })

    # Convert results to DataFrames
    known_df = pd.DataFrame(known_results)
    unknown_df = pd.DataFrame(unknown_results)

    return known_df, unknown_df


def convert_df_to_csv(df):
    """
    Convert a DataFrame to CSV format for download.

    Args:
        df: pandas DataFrame to convert

    Returns:
        bytes: CSV data as bytes
    """
    return df.to_csv(index=False).encode('utf-8')


# ============================================================================
# MAIN UI LAYOUT
# ============================================================================

# Application title
st.title("🦇 Bat Acoustic Identification System")
st.markdown("---")

# ============================================================================
# FILE UPLOAD SECTION
# ============================================================================
st.subheader("Upload Audio Files")

# File uploader widget - accepts multiple .wav files
uploaded_files = st.file_uploader(
    "Upload",
    type=['wav'],
    accept_multiple_files=True,
    key='file_uploader'
)

# Display number of uploaded files
if uploaded_files:
    for file in uploaded_files:
        # Check if file already cached in session state
        if file.name not in [f[0] for f in st.session_state["uploaded_files"]]:
            file_name, sampling_rate, audio_data = cache_wav_file(file.name, file.getvalue())
            st.session_state["uploaded_files"].append((file_name, sampling_rate, audio_data))
    st.info(f"📁 {len(uploaded_files)} file(s) uploaded")
    uploaded_files = st.session_state["uploaded_files"]

st.markdown("---")

# ============================================================================
# CLASSIFICATION SECTION
# ============================================================================
# Ensure detectors state exists
if 'detectors' not in st.session_state:
    st.session_state.detectors = []
if 'show_add_detector' not in st.session_state:
    st.session_state.show_add_detector = False
# Species session state
if 'species' not in st.session_state:
    # list of dicts: Abbreviation, LatinName, CommonName
    st.session_state.species = []

if 'show_add_species' not in st.session_state:
    st.session_state.show_add_species = False

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    # Classify button - triggers the ML model processing
    classify_button = st.button("🔍 Classify", type="primary", use_container_width=True)

with col2:
    # Add new sound detector button - now opens a small form
    add_detector_button = st.button("➕ Add New Sound Detector", use_container_width=True)
with col3:
    # Same look/feel as detector button
    add_species_button = st.button("➕ Add New Species", use_container_width=True)

# Handle classify button click
if classify_button:
    if uploaded_files:
        # Show progress spinner while processing
        with st.spinner('Classifying audio files...'):
            # Simulate processing time (remove in production)
            time.sleep(1)

            # Process the uploaded files using the ML model
            known_df, unknown_df = process_audio_files(uploaded_files)

            # Update session state with new results
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

#Detector input form (shown when toggled)
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
        latin = st.text_input("Latin Name *", placeholder="e.g., Myotis lucifugus")
        common = st.text_input("Common Name *", placeholder="e.g., Little brown bat")

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
                # Accept A–Z / 0–9 / _ / - ; keep short like typical codes
                cleaned = abbr.replace('-', '').replace('_', '')
                if not cleaned.isalnum() or len(abbr) > 16:
                    errors.append("Abbreviation must be alphanumeric (dashes/underscores allowed) and ≤ 16 chars.")

            if not latin:
                errors.append("Latin name is required.")
            if not common:
                errors.append("Common name is required.")

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

# Show a compact summary of registered detectors (if any)
if st.session_state.detectors:
    st.caption("**Registered Detectors**")
    st.dataframe(
        pd.DataFrame(st.session_state.detectors),
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
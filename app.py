"""
app2.py — BatLab: Bat Acoustic Identification Application
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


# ============================================================================
# PAGE CONFIGURATION
# Must be the very first Streamlit call in the script.
# ============================================================================
st.set_page_config(page_title="Bat Acoustic Identification", layout="wide")


# ============================================================================
# GLOBAL CSS — LIGHT MODE (Login page)
# ============================================================================
# This block forces a white/light-mode appearance for the login page.
# After login, a separate dark-mode CSS block overrides most of these rules.
# The heavy use of !important is necessary because Streamlit's own styles
# also use !important extensively.
st.markdown("""
<style>

/* Hide the "View fullscreen" button that appears on images/dataframes */
button[title="View fullscreen"] { display: none !important; }
            
/* ── Root / App background ─────────────────────────────────────────────── */
:root, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

.main, .stApp {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

/* Sidebar gets a light grey background to distinguish it from the main area */
[data-testid="stSidebar"] {
    background-color: #F0F2F6 !important;
}

/* ── Text colour defaults ───────────────────────────────────────────────── */
/* All visible text elements default to black on the light background */
p, span, div, h1, h2, h3, h4, h5, h6, label {
    color: #000000 !important;
}

/* ── Header / Toolbar exceptions ───────────────────────────────────────── */
/*
 * The Streamlit header bar is dark regardless of theme, so its buttons and
 * icons need white text/fill to remain visible.
 * The selectors below target the header dropdown menus and toolbar buttons.
 */

/* Dropdown menu items inside the header need white text (dark dropdown bg) */
[data-testid="stHeaderActionElements"] button,
[data-testid="stHeaderActionElements"] span,
[data-testid="stHeaderActionElements"] div,
button[kind="header"],
header button,
header span,
header div[role="menu"] button,
header div[role="menu"] span,
[role="menu"] button,
[role="menu"] span,
[data-baseweb="menu"] button,
[data-baseweb="menu"] span,
.main-menu button,
.main-menu span {
    color: #FFFFFF !important;
}

/* Toolbar buttons (Deploy, menu icon) — white icon + text */
header button,
header button span,
header button div,
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] button span,
[data-testid="stToolbar"] button div,
[data-testid="stToolbar"] svg,
[data-testid="stDecoration"] + div button,
[data-testid="stDecoration"] + div button span {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}

/* Some header role=button elements should remain black (e.g. status widget) */
button[data-testid*="stHeader"],
button[class*="viewerBadge"],
[data-testid="stStatusWidget"] button,
header [role="button"],
header [role="button"] span,
header [role="button"] svg,
header [role="button"] path {
    color: #000000 !important;
    fill: #000000 !important;
}

/* Strip background/border from header toolbar buttons for a clean look */
header button,
[data-testid="stToolbar"] button,
[data-testid="stDecoration"] + div button {
    background-color: transparent !important;
    border: none !important;
}

/* The "three dots" SVG icon in the header should be white */
header svg,
[data-testid="stToolbar"] svg,
header [role="button"] svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}

/* Subtle circular hover highlight on header buttons */
header button:hover {
    background-color: rgba(0, 0, 0, 0.05) !important;
    border-radius: 50% !important;
}

/* ── Input fields ───────────────────────────────────────────────────────── */
input, textarea, [data-baseweb="input"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #D3D3D3 !important;
    border-radius: 0 !important;
    caret-color: #000000 !important;  /* visible text cursor */
}

/* ── Dialogs / Modals ───────────────────────────────────────────────────── */
/*
 * The Deploy and Settings dialogs keep a dark background (#1E1E1E) for
 * contrast, so all text inside them must be white.
 */
[data-testid="stDialog"] *, 
[role="dialog"] *, 
.st-emotion-cache-12w0qpk * {
    color: #FFFFFF !important;
}

[role="dialog"] h1, 
[role="dialog"] h2, 
[role="dialog"] h3, 
[role="dialog"] p, 
[role="dialog"] li {
    color: #FFFFFF !important;
}

/* Buttons inside dialogs — black bg with white text */
[role="dialog"] button {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    border: 1px solid #FFFFFF !important;
}

/* Hover state for dialog buttons — dark grey with purple border */
[role="dialog"] button:hover {
    background-color: #333333 !important;
    color: #FFFFFF !important;
    border: 1px solid #B19CD9 !important;
}

/* Ensure the button <p> tag text stays white too */
[role="dialog"] button p {
    color: #FFFFFF !important;
}

/* Dark background for all dialog/popover/modal containers */
[data-testid="stDialog"], 
[role="dialog"], 
[data-baseweb="popover"],
[data-baseweb="modal"] {
    background-color: #1E1E1E !important;
}

/* All text nodes inside dialogs must be white (repeated for specificity) */
[role="dialog"] p, 
[role="dialog"] span, 
[role="dialog"] label, 
[role="dialog"] h1, 
[role="dialog"] h2 {
    color: #FFFFFF !important;
}

/* Close (X) button SVG inside dialogs */
[role="dialog"] button svg {
    fill: #FFFFFF !important;
}

/* ── Input focus state ──────────────────────────────────────────────────── */
/* Highlight active input with a purple border; remove default box-shadow */
input:focus, textarea:focus, [data-baseweb="input"]:focus {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    caret-color: #000000 !important;
    border: 2px solid #B19CD9 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}

/* More specific selectors for Streamlit's stTextInput wrapper */
.stTextInput input:focus,
[data-testid="stTextInput"] input:focus {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    caret-color: #000000 !important;
    border: 2px solid #B19CD9 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}

/* ── Hide "Press Enter to submit form" tooltip ──────────────────────────── */
[data-testid="InputInstructions"],
.stTextInput [data-testid="InputInstructions"],
div[data-testid="InputInstructions"] {
    display: none !important;
}

/* Alternative selectors covering different Streamlit builds */
input + div[role="alert"],
input + div[class*="instructions"],
[class*="stTextInput"] div[role="alert"] {
    display: none !important;
}

/* ── Form labels ────────────────────────────────────────────────────────── */
.stTextInput label,
[data-testid="stTextInput"] label,
label {
    font-weight: bold !important;
    color: #000000 !important;
}

/* Extra-bold widget labels (Username / Password on login form) */
[data-testid="stWidgetLabel"] p {
    font-weight: 800 !important;
}

/* Bold text on form submit button (Login) */
[data-testid="stFormSubmitButton"] button p {
    font-weight: 800 !important;
}

/* Repeated selectors for robustness across Streamlit versions */
[data-testid="stWidgetLabel"] p {
    font-weight: bold !important;
}

button[kind="formSubmit"] p {
    font-weight: bold !important;
}

/* All primary and form-submit buttons get bold text */
button[kind="primary"],
button[kind="formSubmit"],
.stFormSubmitButton button,
.stButton button {
    font-weight: bold !important;
}

/* Bold paragraph text inside any .stButton (e.g. "Save Training Data") */
.stButton > button p {
    font-weight: bold !important;
}

/* ── Dropdowns / Select boxes ───────────────────────────────────────────── */
[data-baseweb="select"], [data-baseweb="popover"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

/* ── Generic buttons (light mode) ──────────────────────────────────────── */
button {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #D3D3D3 !important;
}

/* ── DataFrames / Tables ────────────────────────────────────────────────── */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

/* ── Layout blocks ──────────────────────────────────────────────────────── */
[data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"] {
    background-color: #FFFFFF !important;
}

/* ── Form container ─────────────────────────────────────────────────────── */
[data-testid="stForm"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E0E0E0 !important;
}

/* ── Checkboxes — purple accent colour ──────────────────────────────────── */
[data-testid="stCheckbox"] input[type="checkbox"] {
    accent-color: #B19CD9 !important;
}
[data-baseweb="checkbox"] span {
    background-color: #B19CD9 !important;
    border-color: #B19CD9 !important;
    outline-color: #B19CD9 !important;
}
[data-baseweb="checkbox"] span:hover {
    background-color: #9b7fc7 !important;  /* slightly darker purple on hover */
    border-color: #9b7fc7 !important;
}
[role="checkbox"][aria-checked="true"] {
    background-color: #B19CD9 !important;
    border-color: #B19CD9 !important;
}
[role="checkbox"][aria-checked="true"] span {
    background-color: #B19CD9 !important;
    border-color: #B19CD9 !important;
}

/* ── Tabs (light mode) ──────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    background-color: #F0F2F6 !important;
}
[data-baseweb="tab"] {
    color: #000000 !important;
}

/* ── Expander ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E0E0E0 !important;
}

/* ── Alert / Info / Warning / Error boxes (light mode) ─────────────────── */
[data-testid="stAlert"] {
    background-color: #F0F2F6 !important;
    color: #000000 !important;
}

/* ── Metric widgets ─────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: #F0F2F6 !important;
    color: #000000 !important;
}

/* Headings remain black even if a parent rule would make them something else */
h1, h2, h3, h4, h5, h6 {
    color: #000000 !important;
}

/* Utility class for muted / secondary text */
.gray-text {
    color: #666666 !important;
}

/* ── Hide Streamlit hamburger menu & header bar on login page ───────────── */
header[data-testid="stHeader"] {
    display: none !important;
}

/* Remove the default top padding so content starts at the very top */
[data-testid="stAppViewContainer"] {
    padding-top: 0rem !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# ADDITIONAL CSS — Hide unwanted DataFrame toolbar buttons
# ============================================================================
# Streamlit's DataFrame widget shows toolbar buttons (fullscreen, columns,
# download). We only want to hide the "Show/hide columns" and "Download as CSV"
# buttons to keep the interface clean while keeping the search functionality.

# Hide "Show/hide columns" button
st.markdown("""
<style>
/* Most Streamlit builds */
button[title="Show/hide columns"],
button[aria-label="Show/hide columns"] { display: none !important; }

/* Scoped to stDataFrame and stDataEditor (newer testids) */
div[data-testid="stDataFrame"] button[title="Show/hide columns"],
div[data-testid="stDataFrame"] button[aria-label="Show/hide columns"],
div[data-testid="stDataEditor"] button[title="Show/hide columns"],
div[data-testid="stDataEditor"] button[aria-label="Show/hide columns"] { display: none !important; }

/* Fallback via toolbar wrapper */
div[data-testid="stElementToolbar"] button[title*="Show/hide"],
div[data-testid="stElementToolbar"] button[aria-label*="Show/hide"] { display: none !important; }

/* Hide the dialog/menu that opens when the button is clicked */
div[role="dialog"][aria-label="Show/hide columns"],
div[role="menu"][aria-label="Show/hide columns"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# Hide "Download as CSV" button
st.markdown("""
<style>
/* Most Streamlit builds */
button[title="Download as CSV"],
button[aria-label="Download as CSV"] { display:none !important; }

/* Scoped to stDataFrame toolbar */
div[data-testid="stDataFrame"] button[title="Download as CSV"],
div[data-testid="stDataFrame"] button[aria-label="Download as CSV"] { display:none !important; }

/* Fallback via toolbar wrapper */
div[data-testid="stElementToolbar"] button[title*="Download"],
div[data-testid="stElementToolbar"] button[aria-label*="Download"] { display:none !important; }

/* Extra selectors covering both title-case and lowercase attribute values */
div[data-testid="stDataFrame"] [data-testid="stElementToolbar"] button[title*="Download"],
div[data-testid="stDataFrame"] [data-testid="stElementToolbar"] button[aria-label*="Download"],
div[data-testid="stDataFrame"] [data-testid="stElementToolbar"] button[title*="download"],
div[data-testid="stDataFrame"] [data-testid="stElementToolbar"] button[aria-label*="download"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# ADDITIONAL CSS — Button & Tab styling (shared across all pages)
# ============================================================================
# All action buttons (Classify, Save, Cancel, etc.) are styled as transparent
# with a subtle border. On hover/focus they gain a purple (#B19CD9) border.
# The active tab and the tab highlight bar are also purple.
st.markdown("""
<style>
/* ── Action buttons — transparent with light border ────────────────────── */
button[kind="primary"],
button[kind="formSubmit"],
.stFormSubmitButton button,
.stButton button {
    background-color: transparent !important;
    color: #000000 !important;
    border: 1px solid rgba(0, 0, 0, 0.2) !important;
}

/* Hover: swap to purple border */
button[kind="primary"]:hover,
button[kind="formSubmit"]:hover,
.stFormSubmitButton button:hover,
.stButton button:hover {
    background-color: transparent !important;
    color: #000000 !important;
    border: 2px solid #B19CD9 !important;
}

/* Focus and active states also use the purple border */
button[kind="primary"]:focus,
button[kind="formSubmit"]:focus,
.stFormSubmitButton button:focus,
.stButton button:focus {
    border: 2px solid #B19CD9 !important;
    box-shadow: none !important;
}
button[kind="primary"]:active,
button[kind="formSubmit"]:active,
.stFormSubmitButton button:active,
.stButton button:active {
    border: 2px solid #B19CD9 !important;
}

/* ── Tabs — active tab and highlight bar use purple ─────────────────────── */
button[data-baseweb="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"],
div[data-baseweb="tab-list"] button[aria-selected="true"] {
    border-bottom-color: #B19CD9 !important;
    color: #B19CD9 !important;
}

/* The sliding underline/highlight bar beneath the active tab */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #B19CD9 !important;
}

/* Active tab text colour */
.stTabs button[aria-selected="true"] p {
    color: #B19CD9 !important;
}

/* Hover colour on non-active tabs */
.stTabs button[data-baseweb="tab"]:hover,
button[data-baseweb="tab"]:hover,
div[data-baseweb="tab-list"] button:hover {
    color: #B19CD9 !important;
}

.stTabs button[data-baseweb="tab"]:hover p,
button[data-baseweb="tab"]:hover p,
div[data-baseweb="tab-list"] button:hover p {
    color: #B19CD9 !important;
}

/* Catch any nested children inside hovered tab buttons */
.stTabs button:hover *,
button[data-baseweb="tab"]:hover * {
    color: #B19CD9 !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
# Streamlit re-runs the entire script on every user interaction.
# We use st.session_state to persist data between re-runs.
# Each key is only initialised once (using `not in` guard) so that existing
# data is never accidentally overwritten during a re-run.

# --- Classification results ---
# known_data: DataFrame of files where the model confidence met the threshold
if 'known_data' not in st.session_state:
    st.session_state.known_data = pd.DataFrame(
        columns=['Filename', 'Species Prediction', 'Confidence Level']
    )

# unknown_data: DataFrame of files that fell below the confidence threshold
if 'unknown_data' not in st.session_state:
    st.session_state.unknown_data = pd.DataFrame(columns=['Filename'])

# List of .wav filenames discovered in the verified source folder
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# --- Training data ---
# Each entry is a dict: {Species, Detector, FileCount, FileNames}
if 'training_entries' not in st.session_state:
    st.session_state.training_entries = []

# Raw bytes for each uploaded training file, keyed by filename.
# Bytes are accumulated across multiple submissions so they aren't lost.
if 'training_file_bytes' not in st.session_state:
    st.session_state.training_file_bytes = {}

# --- Detectors ---
# Pre-seeded with example South-African detector locations.
# Each entry is a dict: {Detector ID, Latitude, Longitude}
if 'detectors' not in st.session_state:
    st.session_state.detectors = [
        {"Detector ID": "DET-KZN01", "Latitude": "-29.8587", "Longitude": "31.0218"},
        {"Detector ID": "DET-WC02",  "Latitude": "-33.9249", "Longitude": "18.4241"},
        {"Detector ID": "DET-GP03",  "Latitude": "-26.2041", "Longitude": "28.0473"},
        {"Detector ID": "DET-LP04",  "Latitude": "-23.8962", "Longitude": "29.4486"},
        {"Detector ID": "DET-EC05",  "Latitude": "-33.0153", "Longitude": "27.9116"},
        {"Detector ID": "DET-MP06",  "Latitude": "-25.4753", "Longitude": "30.9694"},
        {"Detector ID": "DET-NC07",  "Latitude": "-28.7282", "Longitude": "24.7499"},
        {"Detector ID": "DET-NW08",  "Latitude": "-25.8553", "Longitude": "25.6415"},
        {"Detector ID": "DET-FS09",  "Latitude": "-29.1217", "Longitude": "26.2141"},
    ]

# --- Species ---
# Pre-seeded with common South-African bat species.
# Each entry is a dict: {Abbreviation, Latin Name, Common Name}
if 'species' not in st.session_state:
    st.session_state.species = [
        {"Abbreviation": "TAPMAU", "Latin Name": "Taphozous mauritianus",   "Common Name": "Mauritian Tomb Bat"},
        {"Abbreviation": "TADAEG", "Latin Name": "Tadarida aegyptiaca",     "Common Name": "Egyptian Free-tailed Bat"},
        {"Abbreviation": "OTOMAR", "Latin Name": "Otomops martiensseni",    "Common Name": "Large-eared Free-tailed Bat"},
        {"Abbreviation": "SCODIN", "Latin Name": "Scotophilus dinganii",    "Common Name": "African Yellow Bat"},
        {"Abbreviation": "MINNAT", "Latin Name": "Miniopterus natalensis",  "Common Name": "Natal Long-fingered Bat"},
        {"Abbreviation": "NEOCAP", "Latin Name": "Neoromicia capensis",     "Common Name": "Cape Serotine Bat"},
        {"Abbreviation": "MYOTRI", "Latin Name": "Myotis tricolor",         "Common Name": "Temminck's Myotis"},
        {"Abbreviation": "NYCTHE", "Latin Name": "Nycteris thebaica",       "Common Name": "Egyptian Slit-faced Bat"},
        {"Abbreviation": "RHICAP", "Latin Name": "Rhinolophus capensis",    "Common Name": "Cape Horseshoe Bat"},
    ]

# --- Authentication ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- Uploader keys ---
# Incrementing these keys forces Streamlit to re-render the file uploader
# widgets with an empty state after a successful submission.
if 'training_uploader_key' not in st.session_state:
    st.session_state.training_uploader_key = 0

if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0

# --- Classify tab state ---
# Path of the folder the user entered and verified as containing .wav files
if 'source_folder' not in st.session_state:
    st.session_state.source_folder = ""

# Legacy result message placeholder (currently unused but kept for compatibility)
if 'org_result_msg' not in st.session_state:
    st.session_state.org_result_msg = None

# Controls whether the "move identified files" inline form is visible
if 'show_id_folder_form' not in st.session_state:
    st.session_state.show_id_folder_form = False

# Controls whether the "move unknown files" inline form is visible
if 'show_unk_folder_form' not in st.session_state:
    st.session_state.show_unk_folder_form = False

# Stores the result dict from the last "move identified files" operation
if 'id_folder_result' not in st.session_state:
    st.session_state.id_folder_result = None

# Stores the result dict from the last "move unknown files" operation
if 'unk_folder_result' not in st.session_state:
    st.session_state.unk_folder_result = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def process_audio_files(wav_filenames):
    """
    Simulate ML classification of a list of .wav filenames.

    For each filename a random confidence score is generated.
    Files with confidence >= 0.75 are classified as KNOWN (identified species);
    files below the threshold are classified as UNKNOWN.

    NOTE: This is a placeholder. In production this would call the actual
    trained acoustic model with the audio data.

    Parameters
    ----------
    wav_filenames : list[str]
        Filenames of the .wav files to classify (no bytes are read here).

    Returns
    -------
    known_df : pd.DataFrame
        Columns: Filename, Species Prediction, Confidence Level
    unknown_df : pd.DataFrame
        Columns: Filename
    """
    import random

    known_results   = []
    unknown_results = []

    # Hard-coded placeholder species pool used until a real model is integrated
    placeholder_species = [
        'Myotis lucifugus',
        'Eptesicus fuscus',
        'Lasiurus borealis',
        'Myotis septentrionalis',
        'Perimyotis subflavus',
    ]

    confidence_threshold = 0.75  # Files below this score go to the UNKNOWN table

    for filename in wav_filenames:
        confidence = random.uniform(0.5, 0.99)

        if confidence >= confidence_threshold:
            # Sufficient confidence — pick a random species as the prediction
            predicted_species = random.choice(placeholder_species)
            known_results.append({
                'Filename':           filename,
                'Species Prediction': predicted_species,
                'Confidence Level':   f"{confidence * 100:.2f}%",
            })
        else:
            # Low confidence — mark as unknown
            unknown_results.append({'Filename': filename})

    known_df   = pd.DataFrame(known_results)
    unknown_df = pd.DataFrame(unknown_results)
    return known_df, unknown_df


def convert_df_to_csv(df):
    """
    Encode a DataFrame as a UTF-8 CSV byte string suitable for st.download_button.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    bytes
    """
    return df.to_csv(index=False).encode('utf-8')


def organise_files(source_folder, identified_folder, unknown_folder,
                   known_filenames, unknown_filenames):
    """
    Move classified .wav files from the source folder into separate
    identified and unknown destination folders.

    Files are MOVED (not copied) to avoid duplication. Destination folders
    are created automatically if they do not already exist.

    Parameters
    ----------
    source_folder      : str  — Path that was verified in Step 1 of the Classify tab.
    identified_folder  : str  — Destination for files with a confident species prediction.
    unknown_folder     : str  — Destination for files whose confidence was too low.
    known_filenames    : list[str]  — Filenames from the identified species table.
    unknown_filenames  : list[str]  — Filenames from the unknown species table.

    Returns
    -------
    moved_identified : list[str]  — Successfully moved identified files.
    moved_unknown    : list[str]  — Successfully moved unknown files.
    errors           : list[str]  — Error messages for any files that could not be moved.
    """
    import os, shutil

    moved_identified = []
    moved_unknown    = []
    errors           = []

    def ensure_dir(path):
        """Create a directory (and any missing parents). Return True on success."""
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as exc:
            errors.append(f"Could not create folder '{path}': {exc}")
            return False

    # Validate source folder exists before attempting anything
    if not os.path.isdir(source_folder):
        errors.append(f"Source folder not found: '{source_folder}'")
        return moved_identified, moved_unknown, errors

    # Ensure both destination folders exist (create them if necessary)
    if not ensure_dir(identified_folder):
        return moved_identified, moved_unknown, errors
    if not ensure_dir(unknown_folder):
        return moved_identified, moved_unknown, errors

    def move_files(filenames, dest_folder, moved_list):
        """
        Move each file in `filenames` from source_folder into dest_folder.
        Appends results to moved_list and errors.
        """
        for fname in filenames:
            src = os.path.join(source_folder, fname)
            dst = os.path.join(dest_folder, fname)

            if not os.path.isfile(src):
                # File may have been moved in a previous run
                if os.path.isfile(dst):
                    moved_list.append(f"{fname} (already in destination)")
                else:
                    errors.append(f"File not found in source: '{fname}'")
                continue

            # Guard against source == destination (would silently delete the file)
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
# LOGIN PAGE
# ============================================================================
# If the user is not yet authenticated we show a centred login form and then
# call st.stop() to prevent any of the main app UI from rendering.
if not st.session_state.logged_in:

    # Add vertical whitespace to visually centre the form on the page
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    # Use a 3-column layout so the form appears in the centre column
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # App logo
        st.image("batlablogo.PNG", use_column_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Page heading
        st.markdown(
            "<h2 style='text-align: center; color: #000000; font-weight: bold;'>Please Login</h2>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Login form — wrapped in st.form so both fields submit together
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")

            login_button = st.form_submit_button("Login", use_container_width=True)

            if login_button:
                # NOTE: Hard-coded credentials. Replace with a proper auth
                # mechanism (e.g. hashed passwords in a database) for production.
                if username == "admin" and password == "password":
                    st.session_state.logged_in = True
                    st.success("Login successful!")
                    time.sleep(0.5)   # brief pause so the success message is visible
                    st.rerun()
                else:
                    st.error("Invalid login credentials. Please try again.")

    # Stop script execution here — nothing below this line renders for logged-out users
    st.stop()


# ============================================================================
# MAIN APPLICATION — only reached after successful login
# ============================================================================

if st.session_state.logged_in:
    # --------------------------------------------------------------------------
    # DARK MODE CSS (Main App)
    # --------------------------------------------------------------------------
    # After login the app switches to a black (#000000) background with white
    # text and purple (#B19CD9) accent colour for tabs, radio buttons, etc.
    st.markdown("""
        <style>
            
        /* ── Equal-width tabs ─────────────────────────────────────────────── */
        div[data-baseweb="tab-list"] {
            display: flex !important;
            width: 100% !important;
        }

        /* Each tab stretches to fill an equal share of the tab bar */
        div[data-baseweb="tab-list"] button {
            flex: 1 !important;
            justify-content: center !important;
        }

        /* ── Tab text style ───────────────────────────────────────────────── */
        button[data-baseweb="tab"] {
            font-size: 18px !important;
            font-weight: 700 !important;
            text-align: center !important;
            padding: 12px 0px !important;
        }

        /* Inner <p> tag inside tab buttons — match parent font */
        button[data-baseweb="tab"] p {
            font-size: 18px !important;
            font-weight: 700 !important;
        }

        /* Small gap between tabs */
        div[data-baseweb="tab-list"] {
            gap: 8px !important;
        }
            
        /* ── Background — force black across the entire app ──────────────── */
        :root {
            background-color: #000000 !important;
        }
        body {
            background-color: #000000 !important;
        }
        .main, .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background-color: #000000 !important;
        }

        /* Containers and blocks use transparent bg so the black root shows through */
        div, section, [data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"] {
            background-color: transparent !important;
        }

        /* Main content block */
        .main .block-container {
            background-color: #000000 !important;
        }

        /* ── Text — force white ───────────────────────────────────────────── */
        p, span, div, h1, h2, h3, h4, h5, h6, label, li {
            color: #FFFFFF !important;
        }

        /* ── Sidebar ──────────────────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background-color: #000000 !important;
        }

        /* ── Tabs (dark mode) ─────────────────────────────────────────────── */
        /* Inactive tabs: purple text, no border */
        button[data-baseweb="tab"] {
            color: #B19CD9 !important;
            background-color: transparent !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
        }

        /* Active tab: purple text with a bottom border underline */
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #B19CD9 !important;
            border-bottom-color: #B19CD9 !important;
            background-color: transparent !important;
            border: none !important;
            border-bottom: 2px solid #B19CD9 !important;
            outline: none !important;
            box-shadow: none !important;
        }

        /* Tab bar background */
        [data-baseweb="tab-list"] {
            background-color: #000000 !important;
        }

        /* Remove focus ring on tabs (already indicated by the bottom border) */
        button[data-baseweb="tab"]:focus {
            outline: none !important;
            box-shadow: none !important;
        }

        /* ── DataFrames / Tables (dark mode) ─────────────────────────────── */
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            background-color: #000000 !important;
            color: #FFFFFF !important;
        }

        /* ── Input fields (dark mode) ─────────────────────────────────────── */
        input, textarea, [data-baseweb="input"] {
            background-color: #1A1A1A !important;  /* very dark grey */
            color: #FFFFFF !important;
            border: 1px solid #444444 !important;
            caret-color: #FFFFFF !important;
        }

        /* ── Forms ────────────────────────────────────────────────────────── */
        [data-testid="stForm"] {
            background-color: #000000 !important;
            border: 1px solid #333333 !important;
        }

        /* ── File uploader ────────────────────────────────────────────────── */
        [data-testid="stFileUploader"] {
            background-color: #1A1A1A !important;
        }

        /* "Browse files" button inside the uploader — black bg, white text */
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploaderDropzone"] button {
            background-color: #000000 !important;
            color: #FFFFFF !important;
            border: 1px solid #FFFFFF !important;
        }

        /* Hover: switch border to purple */
        [data-testid="stFileUploader"] button:hover,
        [data-testid="stFileUploaderDropzone"] button:hover {
            background-color: #000000 !important;
            color: #FFFFFF !important;
            border: 2px solid #B19CD9 !important;
        }

        /* ── Radio buttons ────────────────────────────────────────────────── */
        [role="radiogroup"] {
            background-color: #000000 !important;
        }

        /* Override browser default radio appearance with a custom white circle */
        input[type="radio"],
        [data-testid="stRadio"] input[type="radio"],
        [role="radiogroup"] input[type="radio"] {
            appearance: none !important;
            -webkit-appearance: none !important;
            -moz-appearance: none !important;
            width: 18px !important;
            height: 18px !important;
            min-width: 18px !important;
            min-height: 18px !important;
            border: 2px solid #FFFFFF !important;
            border-radius: 50% !important;
            margin-right: 10px !important;
            cursor: pointer !important;
            background-color: transparent !important;
            position: relative !important;
            flex-shrink: 0 !important;
            display: inline-block !important;
        }

        /* Checked radio: filled with purple */
        input[type="radio"]:checked,
        [data-testid="stRadio"] input[type="radio"]:checked,
        [role="radiogroup"] input[type="radio"]:checked {
            background-color: #B19CD9 !important;
            border: 2px solid #FFFFFF !important;
        }

        /* Hover: purple border to hint that the option is clickable */
        input[type="radio"]:hover,
        [data-testid="stRadio"] input[type="radio"]:hover {
            border-color: #B19CD9 !important;
        }

        /* Radio labels — flex layout keeps the circle and text aligned */
        [role="radiogroup"] label,
        [data-testid="stRadio"] label {
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
            padding: 8px 0 !important;
            color: #FFFFFF !important;
        }

        /* Span elements inside radio groups (Streamlit's custom elements) */
        [role="radiogroup"] span,
        [data-testid="stRadio"] span {
            color: #FFFFFF !important;
        }

        /* ── All buttons — purple border on hover ─────────────────────────── */
        button:hover {
            border: 2px solid #B19CD9 !important;
        }
        </style>
    """, unsafe_allow_html=True)


# ============================================================================
# APP HEADER
# ============================================================================
st.title("Welcome to the BatLab!")
st.markdown("Analyze bat acoustic calls and identify species using machine learning")
st.markdown("---")


# ============================================================================
# TAB LAYOUT
# ============================================================================
# The app is organised into 5 tabs. Each tab renders its own section below.
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Classify",
    "Add Detector",
    "Add Species",
    "Add Training Data",
    "Train New Model",
])


# ============================================================================
# TAB 1: CLASSIFY
# ============================================================================
# Workflow:
#   Step 1 — User enters a local folder path containing .wav files.
#   Step 2 — User clicks "Classify" to run the audio classification.
#   Step 3 — Results are shown in two tables:
#             • Identified Species (confidence >= threshold)
#             • Unknown Species    (confidence <  threshold)
#   Each table has an optional "Create Folder & Move Files" action so the
#   user can physically sort the files on disk.
with tab1:

    # ── Tab-specific CSS overrides ───────────────────────────────────────────
    # Alert boxes in this tab use a dark-purple background instead of the
    # default light-grey, to remain readable on the dark app background.
    st.markdown("""
    <style>
                
    /* Alert boxes (info, warning, success, error) — dark purple background */
    div[data-testid="stAlert"] {
        background-color: #2b0a3d !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
    }
    div[data-testid="stAlert"] p {
        color: #FFFFFF !important;
    }
                
    /* File uploader "Browse files" button — purple text and border on hover */
    [data-testid="stFileUploader"] button:hover {
        color: #B19CD9 !important;
        border: 2px solid #B19CD9 !important;
    }
    [data-testid="stFileUploader"] button:hover * {
        color: #B19CD9 !important;
    }

    /* Download ZIP button — solid black background with white text */
    [data-testid="stDownloadButton"] button {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #000000 !important;
    }
    [data-testid="stDownloadButton"] button p {
        color: #FFFFFF !important;
    }
    /* Hover: darken slightly and add purple border */
    [data-testid="stDownloadButton"] button:hover {
        background-color: #222222 !important;
        color: #FFFFFF !important;
        border: 2px solid #B19CD9 !important;
    }
    [data-testid="stDownloadButton"] button:hover p {
        color: #FFFFFF !important;
    }
                
    </style>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.header("Classify Bat Acoustic Calls")

    # ── "Start New Session" button ───────────────────────────────────────────
    # Only shown when there are existing results so the user can reset cleanly.
    if not st.session_state.known_data.empty or not st.session_state.unknown_data.empty:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("**Start New Session**", use_container_width=True):
                # Clear all classification-related state
                st.session_state.known_data    = pd.DataFrame(columns=['Filename', 'Species Prediction', 'Confidence Level'])
                st.session_state.unknown_data  = pd.DataFrame(columns=['Filename'])
                st.session_state.uploaded_files = []
                st.session_state.org_result_msg = None
                st.session_state.source_folder  = ""
                st.session_state.show_id_folder_form  = False
                st.session_state.show_unk_folder_form = False
                st.session_state.id_folder_result     = None
                st.session_state.unk_folder_result    = None
                # Bump the key to force the file-uploader widget to reset
                st.session_state.file_uploader_key += 1
                st.success("✅ Session reset! Ready for new files.")
                st.rerun()

    st.markdown("---")

    # ── STEP 1: Source folder path ───────────────────────────────────────────
    st.markdown("### Step 1 — Enter the source folder path")

    import os
    source_input = st.text_input(
        "Source folder path *",
        value=st.session_state.source_folder,
        placeholder=r"e.g.  /Users/you/BatRecordings  or  C:\BatRecordings",
        key="source_folder_input",
    )

    if st.button("**Verify Path & Load Files**", use_container_width=True):
        source_input = (source_input or "").strip()

        if not source_input:
            st.error("Please enter a folder path.")
        elif not os.path.isdir(source_input):
            st.error(f"Path not found or is not a folder: '{source_input}'")
        else:
            # Scan the folder for .wav files (case-insensitive extension check)
            wav_files = [f for f in os.listdir(source_input) if f.lower().endswith('.wav')]

            if not wav_files:
                st.warning(f"No .wav files found in '{source_input}'.")
            else:
                # Store verified path and file list; reset any previous results
                st.session_state.source_folder  = source_input
                st.session_state.uploaded_files = wav_files
                st.session_state.known_data     = pd.DataFrame(columns=['Filename', 'Species Prediction', 'Confidence Level'])
                st.session_state.unknown_data   = pd.DataFrame(columns=['Filename'])
                st.session_state.id_folder_result  = None
                st.session_state.unk_folder_result = None
                st.success(f"✅ Found {len(wav_files)} .wav file(s) in '{source_input}'.")
                st.rerun()

    # Show a summary of the verified source folder if one has been set
    if st.session_state.source_folder and st.session_state.uploaded_files:
        st.info(
            f"**Source:** `{st.session_state.source_folder}` — "
            f"**{len(st.session_state.uploaded_files)}** .wav file(s) ready for classification."
        )

    st.markdown("---")

    # ── STEP 2: Classify ─────────────────────────────────────────────────────
    st.markdown("### Step 2 — Classify")

    # Disable the button until a valid source folder with files has been verified
    classify_disabled = not bool(st.session_state.source_folder and st.session_state.uploaded_files)

    with st.form("classify_form"):
        submitted = st.form_submit_button(
            "Classify",
            use_container_width=True,
            disabled=classify_disabled,
        )

        if submitted and not classify_disabled:
            with st.spinner("Analyzing audio files… Please wait."):
                time.sleep(2)  # Simulated processing delay
                known_df, unknown_df = process_audio_files(st.session_state.uploaded_files)

                # Append new results to any existing data (supports incremental runs)
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

    # ── IDENTIFIED SPECIES TABLE ─────────────────────────────────────────────
    if not st.session_state.known_data.empty:
        known_count = len(st.session_state.known_data)
        st.subheader(f"😄 Identified Species — {known_count} file{'s' if known_count != 1 else ''}")

        st.dataframe(
            st.session_state.known_data,
            use_container_width=True,
            hide_index=True,
            height=400,
            column_config={
                "Filename":           st.column_config.TextColumn("Filename"),
                "Species Prediction": st.column_config.TextColumn("Species Prediction"),
                "Confidence Level":   st.column_config.TextColumn("Confidence Level"),
            },
        )

        # ── Move identified files to a folder ────────────────────────────────
        # Toggling this button shows/hides the inline form for entering a path.
        if st.button("**Create New Folder for Identified Sound Files**",
                     key="btn_id_folder", use_container_width=True):
            st.session_state.show_id_folder_form = not st.session_state.show_id_folder_form
            st.session_state.id_folder_result    = None  # clear any previous result

        if st.session_state.show_id_folder_form:
            with st.form("id_folder_form"):
                st.markdown("**Enter the full path for the new identified species folder**")
                st.markdown(
                    "The folder will be created if it does not exist, and files will be "
                    "**moved** from the source folder."
                )
                id_folder_path = st.text_input(
                    "Destination folder path *",
                    placeholder=r"e.g.  /Users/you/Identified_Bats  or  C:\Identified_Bats",
                )
                col_save_id, col_cancel_id = st.columns(2)
                with col_save_id:
                    confirm_id = st.form_submit_button("**Create Folder & Move Files**", use_container_width=True)
                with col_cancel_id:
                    cancel_id  = st.form_submit_button("**Cancel**", use_container_width=True)

                if cancel_id:
                    # User cancelled — hide the form
                    st.session_state.show_id_folder_form = False
                    st.rerun()

                if confirm_id:
                    import os, shutil
                    id_folder_path = (id_folder_path or "").strip()

                    if not id_folder_path:
                        st.error("Please enter a destination folder path.")
                    elif os.path.abspath(id_folder_path) == os.path.abspath(st.session_state.source_folder):
                        # Prevent moving files back into their source folder
                        st.error("Destination must be different from the source folder.")
                    else:
                        known_names = st.session_state.known_data['Filename'].tolist()

                        # Create the destination folder
                        try:
                            os.makedirs(id_folder_path, exist_ok=True)
                        except Exception as exc:
                            st.error(f"Could not create folder: {exc}")
                            st.stop()

                        # Move each identified file one by one
                        moved, failed = [], []
                        for fname in known_names:
                            src = os.path.join(st.session_state.source_folder, fname)
                            dst = os.path.join(id_folder_path, fname)
                            if not os.path.isfile(src):
                                failed.append(f"{fname} (not found in source)")
                                continue
                            try:
                                shutil.move(src, dst)
                                moved.append(fname)
                            except Exception as exc:
                                failed.append(f"{fname} ({exc})")

                        # Store the result so it persists after st.rerun()
                        st.session_state.id_folder_result = {
                            "path":   id_folder_path,
                            "moved":  moved,
                            "failed": failed,
                        }
                        st.session_state.show_id_folder_form = False
                        st.rerun()

        # Show the result of the last "move identified files" operation
        if st.session_state.id_folder_result:
            res = st.session_state.id_folder_result
            if res["moved"]:
                st.success(f"Moved **{len(res['moved'])}** file(s) to `{res['path']}`.")
            if res["failed"]:
                st.warning(
                    "The following files could not be moved:\n"
                    + "\n".join(f"• {f}" for f in res["failed"])
                )

    else:
        # No results yet — prompt the user to run the classifier
        st.info("No identified species yet. Verify a source folder and run classify to get started!")

    st.markdown("---")

    # ── UNKNOWN SPECIES TABLE ────────────────────────────────────────────────
    if not st.session_state.unknown_data.empty:
        unknown_count = len(st.session_state.unknown_data)
        st.subheader(f"🤔 Unknown Species — {unknown_count} file{'s' if unknown_count != 1 else ''}")

        st.dataframe(
            st.session_state.unknown_data,
            use_container_width=True,
            hide_index=True,
            height=400,
        )

        # ── Move unknown files to a folder ───────────────────────────────────
        # Same pattern as the identified-files section above.
        if st.button("**Create New Folder for Unknown Sound Files**",
                     key="btn_unk_folder", use_container_width=True):
            st.session_state.show_unk_folder_form = not st.session_state.show_unk_folder_form
            st.session_state.unk_folder_result    = None

        if st.session_state.show_unk_folder_form:
            with st.form("unk_folder_form"):
                st.markdown("**Enter the full path for the new unknown species folder**")
                st.markdown(
                    "The folder will be created if it does not exist, and files will be "
                    "**moved** from the source folder."
                )
                unk_folder_path = st.text_input(
                    "Destination folder path *",
                    placeholder=r"e.g.  /Users/you/Unknown_Bats  or  C:\Unknown_Bats",
                )
                col_save_unk, col_cancel_unk = st.columns(2)
                with col_save_unk:
                    confirm_unk = st.form_submit_button("**Create Folder & Move Files**", use_container_width=True)
                with col_cancel_unk:
                    cancel_unk  = st.form_submit_button("**Cancel**", use_container_width=True)

                if cancel_unk:
                    st.session_state.show_unk_folder_form = False
                    st.rerun()

                if confirm_unk:
                    import os, shutil
                    unk_folder_path = (unk_folder_path or "").strip()

                    if not unk_folder_path:
                        st.error("Please enter a destination folder path.")
                    elif os.path.abspath(unk_folder_path) == os.path.abspath(st.session_state.source_folder):
                        st.error("Destination must be different from the source folder.")
                    else:
                        unknown_names = st.session_state.unknown_data['Filename'].tolist()

                        # Create the destination folder
                        try:
                            os.makedirs(unk_folder_path, exist_ok=True)
                        except Exception as exc:
                            st.error(f"Could not create folder: {exc}")
                            st.stop()

                        # Move each unknown file one by one
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

                        # Persist result across re-run
                        st.session_state.unk_folder_result = {
                            "path":   unk_folder_path,
                            "moved":  moved,
                            "failed": failed,
                        }
                        st.session_state.show_unk_folder_form = False
                        st.rerun()

        # Show the result of the last "move unknown files" operation
        if st.session_state.unk_folder_result:
            res = st.session_state.unk_folder_result
            if res["moved"]:
                st.success(f"Moved **{len(res['moved'])}** file(s) to `{res['path']}`.")
            if res["failed"]:
                st.warning(
                    "The following files could not be moved:\n"
                    + "\n".join(f"• {f}" for f in res["failed"])
                )

    st.markdown("---")


# ============================================================================
# TAB 2: ADD DETECTOR
# ============================================================================
# Allows users to register new acoustic detector units.
# Each detector requires a unique ID and valid GPS coordinates.
# Registered detectors are stored in session_state and displayed in a table.
with tab2:
    st.markdown("---")
    st.header("Register a New Detector")
    st.markdown("---")

    with st.form("add_detector_form", clear_on_submit=False):
        name = st.text_input("Detector ID *", placeholder="e.g., Detector-A1")
        lat  = st.text_input("Latitude *",    placeholder="e.g., -33.9249")
        lon  = st.text_input("Longitude *",   placeholder="e.g., 18.4241")

        submitted_detector = st.form_submit_button("Save Detector", use_container_width=True)

        if submitted_detector:
            errors = []

            # Strip whitespace from all inputs
            name = (name or "").strip()
            lat  = (lat  or "").strip()
            lon  = (lon  or "").strip()

            # ── Validation ───────────────────────────────────────────────────
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
                # Check for an exact duplicate (same ID AND same coordinates)
                duplicate = any(
                    d["Detector ID"] == name
                    and float(d["Latitude"])  == float(lat)
                    and float(d["Longitude"]) == float(lon)
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
                # All checks passed — append the new detector to session state
                st.session_state.detectors.append({
                    "Detector ID": name,
                    "Latitude":    lat,
                    "Longitude":   lon,
                })
                st.success("New detector saved successfully!")
                st.rerun()

    st.markdown("---")

    # Display all registered detectors in a scrollable table
    if st.session_state.detectors:
        st.subheader("Registered Detectors")
        st.dataframe(
            pd.DataFrame(st.session_state.detectors),
            use_container_width=True,
            hide_index=True,
            height=400,
        )
    else:
        st.info("No detectors registered yet. Add your first detector above!")


# ============================================================================
# TAB 3: ADD SPECIES
# ============================================================================
# Allows users to register bat species records with an abbreviation code,
# Latin (scientific) name, and optional common name.
# Duplicate entries (same abbreviation + same Latin name) are rejected.
with tab3:
    st.markdown("---")
    st.header("Register a New Species")
    st.markdown("---")

    with st.form("add_species_form", clear_on_submit=False):
        abbr   = st.text_input("Abbreviation *",           placeholder="e.g., MYLU")
        latin  = st.text_input("Latin Name *",             placeholder="e.g., Myotis lucifugus")
        common = st.text_input("Common Name (optional)",   placeholder="e.g., Little brown bat")

        submitted_species = st.form_submit_button("Save Species", use_container_width=True)

        if submitted_species:
            errors = []

            # Strip whitespace
            abbr   = (abbr   or "").strip()
            latin  = (latin  or "").strip()
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
                # Case-insensitive duplicate check on both abbreviation and Latin name
                duplicate = any(
                    s["Abbreviation"].lower() == abbr.lower()
                    and s["Latin Name"].lower() == latin.lower()
                    for s in st.session_state.species
                )
                if duplicate:
                    errors.append(
                        f"A species with abbreviation '{abbr}' and Latin name '{latin}' already exists."
                    )

            if errors:
                for e in errors:
                    st.error(e)
            else:
                # All checks passed — append the new species record
                st.session_state.species.append({
                    "Abbreviation": abbr,
                    "Latin Name":   latin,
                    "Common Name":  common,
                })
                st.success("New species saved successfully!")
                st.rerun()

    st.markdown("---")

    # Display all registered species in a scrollable table
    if st.session_state.species:
        st.subheader("Registered Species")
        species_df = pd.DataFrame(st.session_state.species)
        st.dataframe(
            species_df,
            use_container_width=True,
            hide_index=True,
            height=400,
        )


# ============================================================================
# TAB 4: ADD TRAINING DATA
# ============================================================================
# Allows users to associate .wav audio files with a species and detector.
# This data is intended to be used as labelled training input for model training.
#
# Workflow:
#   1. Select a species (filterable radio list).
#   2. Select a detector (filterable radio list).
#   3. Upload one or more .wav files (max 200 MB each).
#   4. Click "Save Training Data" — files are stored in memory and an entry
#      is recorded in session_state.training_entries.
with tab4:
    st.markdown("---")
    st.header("Add Training Data")
    st.markdown("---")

    col_species, col_location = st.columns(2)

    # ── Species selection column ─────────────────────────────────────────────
    with col_species:
        st.markdown("**Select Species:**")

        # Live-filter the species list as the user types
        species_search = st.text_input(
            "Search species",
            placeholder="Type to filter...",
            key="species_search",
        )

        all_species = [s['Abbreviation'] for s in st.session_state.species]
        filtered_species = (
            [sp for sp in all_species if species_search.lower() in sp.lower()]
            if species_search else all_species
        )

        selected_species = st.radio(
            "Species options",
            options=filtered_species if filtered_species else ["No species registered yet"],
            label_visibility="collapsed",
            key="species_radio",
        )

    # ── Detector selection column ────────────────────────────────────────────
    with col_location:
        st.markdown("**Select Detector:**")

        # Live-filter the detector list as the user types
        detector_search = st.text_input(
            "Search detector",
            placeholder="Type to filter...",
            key="detector_search",
        )

        all_detectors = [d["Detector ID"] for d in st.session_state.detectors]
        filtered_detectors = (
            [d for d in all_detectors if detector_search.lower() in d.lower()]
            if detector_search else all_detectors
        )

        selected_detector = st.radio(
            "Detector options",
            options=filtered_detectors if filtered_detectors else ["No detectors registered yet"],
            label_visibility="collapsed",
            key="detector_radio",
        )

    st.markdown("---")

    # ── File upload ──────────────────────────────────────────────────────────
    st.markdown("**Upload Training Audio Files (.wav, max 200MB per file):**")

    # The uploader key is incremented after a successful save to reset the widget
    training_files = st.file_uploader(
        "Drop .wav files here",
        type=['wav'],
        accept_multiple_files=True,
        key=f'training_file_uploader_{st.session_state.training_uploader_key}',
        label_visibility="collapsed",
    )

    if training_files:
        # Warn the user immediately if any file exceeds the size limit
        oversized_files = [
            f"{f.name} ({f.size / (1024 * 1024):.1f}MB)"
            for f in training_files
            if f.size / (1024 * 1024) > 200
        ]
        if oversized_files:
            st.error(f"The following files exceed 200MB limit: {', '.join(oversized_files)}")
        else:
            st.info(f"{len(training_files)} training file(s) selected")

    st.markdown("---")

    submitted_training = st.button("Save Training Data", use_container_width=True, type="primary")

    if submitted_training:
        errors = []

        # ── Validation ───────────────────────────────────────────────────────
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
            # Read and store the raw bytes for each uploaded file
            new_file_names = []
            for f in training_files:
                file_bytes = f.read()
                st.session_state.training_file_bytes[f.name] = file_bytes  # accumulated store
                new_file_names.append(f.name)

            # Record the training entry (species + detector + file list)
            st.session_state.training_entries.append({
                "Species":   selected_species,
                "Detector":  selected_detector,
                "FileCount": len(training_files),
                "FileNames": new_file_names,
            })

            # Bump the uploader key to clear the file-uploader widget
            st.session_state.training_uploader_key += 1

            st.success(
                f"Training data saved: {len(training_files)} file(s) "
                f"for {selected_species} at detector '{selected_detector}'."
            )
            st.rerun()

    st.markdown("---")

    # Display all saved training entries
    if st.session_state.training_entries:
        st.subheader("Training Data Entries")
        training_df = pd.DataFrame([
            {
                "Species":  entry["Species"],
                # Support both old "Location" key and new "Detector" key for compatibility
                "Detector": entry.get("Detector", entry.get("Location", "")),
                "Files":    entry["FileCount"],
            }
            for entry in st.session_state.training_entries
        ])
        st.dataframe(
            training_df,
            use_container_width=True,
            hide_index=True,
            height=400,
        )
    else:
        st.info("No training data entries yet. Add your first training dataset above!")


# ============================================================================
# TAB 5: TRAIN NEW MODEL
# ============================================================================
# Allows users to configure and submit a model training job.
#
# Workflow:
#   1. User searches/filters the list of registered detectors.
#   2. For each detector they want to include, they tick a checkbox.
#   3. For each selected detector, species checkboxes appear so they can
#      specify which species the model should be trained on.
#   4. A summary is shown. The "Train Model" button submits the job.
#
# NOTE: The actual training call is not yet implemented — the button currently
#       shows a success message as a placeholder.
with tab5:
    st.markdown("---")
    st.header("Train New Model")
    st.markdown("---")

    # Guard: require at least one detector and one species to be registered
    if not st.session_state.detectors:
        st.info("No detectors registered yet. Please add detectors in the 'Add Detector' tab first.")
    elif not st.session_state.species:
        st.info("No species registered yet. Please add species in the 'Add Species' tab first.")
    else:
        st.markdown("Select the detectors and species you want to include in the new model training run.")
        st.markdown("---")

        # Initialise the selections dict if this is the first visit to this tab
        if "train_selections" not in st.session_state:
            st.session_state.train_selections = {}

        all_species_options = [s["Abbreviation"] for s in st.session_state.species]
        all_detector_ids    = [d["Detector ID"]  for d in st.session_state.detectors]

        # ── Detector search/filter ────────────────────────────────────────────
        detector_search = st.text_input(
            "Search detectors",
            placeholder="Type to filter detectors...",
            key="train_detector_search",
        )
        st.markdown("---")

        filtered_detectors = (
            [d for d in all_detector_ids if detector_search.lower() in d.lower()]
            if detector_search else all_detector_ids
        )

        if not filtered_detectors:
            st.info("No detectors match your search.")

        # ── Render one row per (visible) detector ─────────────────────────────
        for det_id in filtered_detectors:
            # Ensure a selection record exists for this detector
            if det_id not in st.session_state.train_selections:
                st.session_state.train_selections[det_id] = {
                    "selected":      False,
                    "species":       [],
                    "species_search": "",
                }

            # Checkbox + label in a 2-column layout (thin left col for the checkbox)
            col_check, col_label = st.columns([0.05, 0.95])
            with col_check:
                selected = st.checkbox(
                    "",
                    value=st.session_state.train_selections[det_id]["selected"],
                    key=f"det_check_{det_id}",
                )
            with col_label:
                st.markdown(f"**{det_id}**")

            # Persist the checkbox state
            st.session_state.train_selections[det_id]["selected"] = selected

            # If this detector is selected, show species checkboxes beneath it
            if selected:
                st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;*Select species:*")
                chosen_species = []
                for sp in all_species_options:
                    already = sp in st.session_state.train_selections[det_id]["species"]
                    checked = st.checkbox(
                        sp,
                        value=already,
                        key=f"sp_check_{det_id}_{sp}",
                    )
                    if checked:
                        chosen_species.append(sp)
                # Update species selection for this detector
                st.session_state.train_selections[det_id]["species"] = chosen_species

            st.markdown("---")

        # ── Training summary & submit button ─────────────────────────────────
        # Collect only the detectors that have been ticked
        active = {
            det: info
            for det, info in st.session_state.train_selections.items()
            if info["selected"]
        }

        if active:
            st.markdown("**Selected for training:**")
            all_valid = True  # becomes False if any selected detector has no species chosen

            for det_id, info in active.items():
                if info["species"]:
                    st.markdown(f"- **{det_id}**: {', '.join(info['species'])}")
                else:
                    st.markdown(f"- **{det_id}**: No species selected")
                    all_valid = False

            st.markdown("")

            if st.button("Train Model", use_container_width=True, type="primary"):
                if not all_valid:
                    # At least one selected detector has no species — block submission
                    st.error(
                        "Please select at least one species for each chosen detector before training."
                    )
                else:
                    # Placeholder success message — replace with actual training API call
                    st.success(
                        f"Training job submitted for {len(active)} detector(s): "
                        + ", ".join(active.keys())
                    )
        else:
            st.info("Select at least one detector above to configure your training run.")

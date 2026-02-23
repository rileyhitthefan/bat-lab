import streamlit as st
import pandas as pd
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(page_title="Bat Acoustic Identification", layout="wide")

# Force light mode - override all dark mode styles
st.markdown("""
<style>

button[title="View fullscreen"] { display: none !important; }
            
/* Force light mode colors */
:root, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

/* Main app background */
.main, .stApp {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #F0F2F6 !important;
}

/* All text elements */
p, span, div, h1, h2, h3, h4, h5, h6, label {
    color: #000000 !important;
}

/* Exception: Menu items in dropdown menus need white text */
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

/* Make header toolbar buttons visible (Deploy button and menu icon) */
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

/* Additional selectors for header buttons */
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

/* Make header toolbar buttons visible with clean icons */
header button,
[data-testid="stToolbar"] button,
[data-testid="stDecoration"] + div button {
    background-color: transparent !important;
    border: none !important;
}

/* Specifically target the SVG icon (the three dots) to be black */
header svg,
[data-testid="stToolbar"] svg,
header [role="button"] svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}

/* Add a subtle circular hover effect */
header button:hover {
    background-color: rgba(0, 0, 0, 0.05) !important;
    border-radius: 50% !important;
}

/* Input fields */
input, textarea, [data-baseweb="input"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #D3D3D3 !important;
    border-radius: 0 !important;
    caret-color: #000000 !important;
}

/* Fix for Deploy Menu and Dialog text visibility */
[data-testid="stDialog"] *, 
[role="dialog"] *, 
.st-emotion-cache-12w0qpk * {
    color: #FFFFFF !important;
}

/* Ensure sub-headers and descriptions are also white */
[role="dialog"] h1, 
[role="dialog"] h2, 
[role="dialog"] h3, 
[role="dialog"] p, 
[role="dialog"] li {
    color: #FFFFFF !important;
}

/* Style buttons inside the Deploy/Settings dialogs */
[role="dialog"] button {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    border: 1px solid #FFFFFF !important;
}

/* Ensure the button text stays white on hover */
[role="dialog"] button:hover {
    background-color: #333333 !important;
    color: #FFFFFF !important;
    border: 1px solid #B19CD9 !important;
}

/* Fix for specific button text elements */
[role="dialog"] button p {
    color: #FFFFFF !important;
}
            
/* Fix for Settings and Dialog visibility */
[data-testid="stDialog"], 
[role="dialog"], 
[data-baseweb="popover"],
[data-baseweb="modal"] {
    background-color: #1E1E1E !important; /* Keep it dark */
}

/* Ensure text inside the dialog is white so it can be read */
[role="dialog"] p, 
[role="dialog"] span, 
[role="dialog"] label, 
[role="dialog"] h1, 
[role="dialog"] h2 {
    color: #FFFFFF !important;
}

/* Make sure the close button (X) is visible */
[role="dialog"] button svg {
    fill: #FFFFFF !important;
}

/* Ensure input fields show cursor on focus */
input:focus, textarea:focus, [data-baseweb="input"]:focus {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    caret-color: #000000 !important;
    border: 2px solid #B19CD9 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}

/* More specific targeting for Streamlit text inputs */
.stTextInput input:focus,
[data-testid="stTextInput"] input:focus {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    caret-color: #000000 !important;
    border: 2px solid #B19CD9 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}

/* Hide the "Press Enter to submit form" tooltip */
[data-testid="InputInstructions"],
.stTextInput [data-testid="InputInstructions"],
div[data-testid="InputInstructions"] {
    display: none !important;
}

/* Alternative selectors for the submit form instruction */
input + div[role="alert"],
input + div[class*="instructions"],
[class*="stTextInput"] div[role="alert"] {
    display: none !important;
}

/* Make form labels bold and black */
.stTextInput label,
[data-testid="stTextInput"] label,
label {
    font-weight: bold !important;
    color: #000000 !important;
}

/* Bold the Username and Password labels */
[data-testid="stWidgetLabel"] p {
    font-weight: 800 !important;
}

/* Bold the Login button text */
[data-testid="stFormSubmitButton"] button p {
    font-weight: 800 !important;
}
            
/* Bold the Username and Password labels */
[data-testid="stWidgetLabel"] p {
    font-weight: bold !important;
}

/* Bold the Login button text */
button[kind="formSubmit"] p {
    font-weight: bold !important;
}

/* Make button text bold */
button[kind="primary"],
button[kind="formSubmit"],
.stFormSubmitButton button,
.stButton button {
    font-weight: bold !important;
}

/* Bold Save Training Data button specifically */
.stButton > button p {
    font-weight: bold !important;
}

/* Select boxes and dropdowns */
[data-baseweb="select"], [data-baseweb="popover"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

/* Buttons - keep transparent with border */
button {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #D3D3D3 !important;
}

/* DataFrames and tables */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

/* Cards and containers */
[data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"] {
    background-color: #FFFFFF !important;
}

/* Forms */
[data-testid="stForm"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E0E0E0 !important;
}

/* Purple checkboxes */
[data-testid="stCheckbox"] input[type="checkbox"] {
    accent-color: #B19CD9 !important;
}
[data-baseweb="checkbox"] span {
    background-color: #B19CD9 !important;
    border-color: #B19CD9 !important;
    outline-color: #B19CD9 !important;
}
[data-baseweb="checkbox"] span:hover {
    background-color: #9b7fc7 !important;
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

/* Tabs */
[data-baseweb="tab-list"] {
    background-color: #F0F2F6 !important;
}

[data-baseweb="tab"] {
    color: #000000 !important;
}

/* Expander */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E0E0E0 !important;
}

/* Info/warning/error boxes */
[data-testid="stAlert"] {
    background-color: #F0F2F6 !important;
    color: #000000 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #F0F2F6 !important;
    color: #000000 !important;
}

/* Headers remain black */
h1, h2, h3, h4, h5, h6 {
    color: #000000 !important;
}

/* Gray text for subtitles */
.gray-text {
    color: #666666 !important;
}

/* Completely hide the hamburger menu and header toolbar */
header[data-testid="stHeader"] {
    display: none !important;
}

/* Remove the top padding to pull the content to the top */
[data-testid="stAppViewContainer"] {
    padding-top: 0rem !important;
}
</style>
""", unsafe_allow_html=True)

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

/* Hide ONLY the download icon in the dataframe toolbar (keep search) */
div[data-testid="stDataFrame"] [data-testid="stElementToolbar"] button[title*="Download"],
div[data-testid="stDataFrame"] [data-testid="stElementToolbar"] button[aria-label*="Download"],
div[data-testid="stDataFrame"] [data-testid="stElementToolbar"] button[title*="download"],
div[data-testid="stDataFrame"] [data-testid="stElementToolbar"] button[aria-label*="download"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# Make Classify button blue and adjust tab styling
st.markdown("""
<style>
/* Target all form submit buttons (Classify, Save Detector, Save Species, Save Training Data, Cancel) */
button[kind="primary"],
button[kind="formSubmit"],
.stFormSubmitButton button,
.stButton button {
    background-color: transparent !important;
    color: #000000 !important;
    border: 1px solid rgba(0, 0, 0, 0.2) !important;
}
button[kind="primary"]:hover,
button[kind="formSubmit"]:hover,
.stFormSubmitButton button:hover,
.stButton button:hover {
    background-color: transparent !important;
    color: #000000 !important;
    border: 2px solid #B19CD9 !important;
}
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

/* Make ALL tabs use purple */
button[data-baseweb="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"],
div[data-baseweb="tab-list"] button[aria-selected="true"] {
    border-bottom-color: #B19CD9 !important;
    color: #B19CD9 !important;
}

/* Target the tab highlight/border directly for all tabs */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #B19CD9 !important;
}

/* Make sure tab text is also purple when selected */
.stTabs button[aria-selected="true"] p {
    color: #B19CD9 !important;
}

/* Change tab hover color to purple - more specific selectors */
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

/* Target any nested elements that might have the purple color */
.stTabs button:hover *,
button[data-baseweb="tab"]:hover * {
    color: #B19CD9 !important;
}
</style>
""", unsafe_allow_html=True)

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
    st.session_state.detectors = [
        {"Detector ID": "DET-KZN01", "Latitude": "-29.8587", "Longitude": "31.0218"},
        {"Detector ID": "DET-WC02", "Latitude": "-33.9249", "Longitude": "18.4241"},
        {"Detector ID": "DET-GP03", "Latitude": "-26.2041", "Longitude": "28.0473"},
        {"Detector ID": "DET-LP04", "Latitude": "-23.8962", "Longitude": "29.4486"},
        {"Detector ID": "DET-EC05", "Latitude": "-33.0153", "Longitude": "27.9116"},
        {"Detector ID": "DET-MP06", "Latitude": "-25.4753", "Longitude": "30.9694"},
        {"Detector ID": "DET-NC07", "Latitude": "-28.7282", "Longitude": "24.7499"},
        {"Detector ID": "DET-NW08", "Latitude": "-25.8553", "Longitude": "25.6415"},
        {"Detector ID": "DET-FS09", "Latitude": "-29.1217", "Longitude": "26.2141"},
    ]

if 'species' not in st.session_state:
    st.session_state.species = [
        {"Abbreviation": "TAPMAU", "Latin Name": "Taphozous mauritianus", "Common Name": "Mauritian Tomb Bat"},
        {"Abbreviation": "TADAEG", "Latin Name": "Tadarida aegyptiaca", "Common Name": "Egyptian Free-tailed Bat"},
        {"Abbreviation": "OTOMAR", "Latin Name": "Otomops martiensseni", "Common Name": "Large-eared Free-tailed Bat"},
        {"Abbreviation": "SCODIN", "Latin Name": "Scotophilus dinganii", "Common Name": "African Yellow Bat"},
        {"Abbreviation": "MINNAT", "Latin Name": "Miniopterus natalensis", "Common Name": "Natal Long-fingered Bat"},
        {"Abbreviation": "NEOCAP", "Latin Name": "Neoromicia capensis", "Common Name": "Cape Serotine Bat"},
        {"Abbreviation": "MYOTRI", "Latin Name": "Myotis tricolor", "Common Name": "Temminck's Myotis"},
        {"Abbreviation": "NYCTHE", "Latin Name": "Nycteris thebaica", "Common Name": "Egyptian Slit-faced Bat"},
        {"Abbreviation": "RHICAP", "Latin Name": "Rhinolophus capensis", "Common Name": "Cape Horseshoe Bat"},
    ]

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

def process_audio_files(wav_filenames):
    """Classify a list of .wav filenames as KNOWN or UNKNOWN (reads no bytes)."""
    import random
    known_results = []
    unknown_results = []

    for filename in wav_filenames:
        confidence = random.uniform(0.5, 0.99)
        confidence_threshold = 0.75

        if confidence >= confidence_threshold:
            species_list = ['Myotis lucifugus', 'Eptesicus fuscus', 'Lasiurus borealis',
                            'Myotis septentrionalis', 'Perimyotis subflavus']
            predicted_species = random.choice(species_list)
            known_results.append({
                'Filename': filename,
                'Species Prediction': predicted_species,
                'Confidence Level': f"{confidence * 100:.2f}%"
            })
        else:
            unknown_results.append({'Filename': filename})

    known_df = pd.DataFrame(known_results)
    unknown_df = pd.DataFrame(unknown_results)
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
        st.image("batlablogo.PNG", use_column_width=True)
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
    st.markdown("""
        <style>
            
            /* ===== EQUAL WIDTH TABS ===== */
            div[data-baseweb="tab-list"] {
            display: flex !important;
            width: 100% !important;
            }

            /* Each tab fills equal space */
            div[data-baseweb="tab-list"] button {
            flex: 1 !important;
            justify-content: center !important;
            }

            /* ===== TAB TEXT STYLE ===== */
            button[data-baseweb="tab"] {
            font-size: 18px !important;
            font-weight: 700 !important;
            text-align: center !important;
            padding: 12px 0px !important;   /* vertical padding */
            }

            /* Ensure inner text matches */
            button[data-baseweb="tab"] p {
            font-size: 18px !important;
            font-weight: 700 !important;
            }

            /* Spacing between tab labels */
            div[data-baseweb="tab-list"] {
            gap: 8px !important;
            }
                
            /* Force Black Background - Multiple selectors for maximum override */
            :root {
                background-color: #000000 !important;
            }
            
            body {
                background-color: #000000 !important;
            }
            
            .main, .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
                background-color: #000000 !important;
            }
            
            /* Force all container backgrounds to black */
            div, section, [data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"] {
                background-color: transparent !important;
            }
            
            /* Main content area */
            .main .block-container {
                background-color: #000000 !important;
            }
            
            /* Force All Text to White */
            p, span, div, h1, h2, h3, h4, h5, h6, label, li {
                color: #FFFFFF !important;
            }

            /* Sidebar Black */
            [data-testid="stSidebar"] {
                background-color: #000000 !important;
            }

            /* Tabs Styling */
            button[data-baseweb="tab"] {
                color: #B19CD9 !important;
                background-color: transparent !important;
                border: none !important;
                outline: none !important;
                box-shadow: none !important;
            }
                
            button[data-baseweb="tab"][aria-selected="true"] {
                color: #B19CD9 !important;
                border-bottom-color: #B19CD9 !important;
                background-color: transparent !important;
                border: none !important;
                border-bottom: 2px solid #B19CD9 !important;
                outline: none !important;
                box-shadow: none !important;
            }
            
            [data-baseweb="tab-list"] {
                background-color: #000000 !important;
            }
            
            /* Remove focus outline on tabs */
            button[data-baseweb="tab"]:focus {
                outline: none !important;
                box-shadow: none !important;
            }

            /* Dataframes/Tables */
            [data-testid="stDataFrame"], [data-testid="stTable"] {
                background-color: #000000 !important;
                color: #FFFFFF !important;
            }

            /* Input boxes and text areas */
            input, textarea, [data-baseweb="input"] {
                background-color: #1A1A1A !important;
                color: #FFFFFF !important;
                border: 1px solid #444444 !important;
                caret-color: #FFFFFF !important;
            }
            
            /* Forms */
            [data-testid="stForm"] {
                background-color: #000000 !important;
                border: 1px solid #333333 !important;
            }
            
            /* File uploader */
            [data-testid="stFileUploader"] {
                background-color: #1A1A1A !important;
            }
            
            /* Browse files button - black background with white text */
            [data-testid="stFileUploader"] button,
            [data-testid="stFileUploaderDropzone"] button {
                background-color: #000000 !important;
                color: #FFFFFF !important;
                border: 1px solid #FFFFFF !important;
            }
            
            [data-testid="stFileUploader"] button:hover,
            [data-testid="stFileUploaderDropzone"] button:hover {
                background-color: #000000 !important;
                color: #FFFFFF !important;
                border: 2px solid #B19CD9 !important;
            }
            
            /* Radio buttons container */
            [role="radiogroup"] {
                background-color: #000000 !important;
            }
            
            /* Streamlit radio button styling - multiple selectors for compatibility */
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
            
            /* Radio button when checked - fill with purple */
            input[type="radio"]:checked,
            [data-testid="stRadio"] input[type="radio"]:checked,
            [role="radiogroup"] input[type="radio"]:checked {
                background-color: #B19CD9 !important;
                border: 2px solid #FFFFFF !important;
            }
            
            /* Radio button on hover */
            input[type="radio"]:hover,
            [data-testid="stRadio"] input[type="radio"]:hover {
                border-color: #B19CD9 !important;
            }
            
            /* Radio button labels */
            [role="radiogroup"] label,
            [data-testid="stRadio"] label {
                cursor: pointer !important;
                display: flex !important;
                align-items: center !important;
                padding: 8px 0 !important;
                color: #FFFFFF !important;
            }
            
            /* Streamlit's custom radio span elements */
            [role="radiogroup"] span,
            [data-testid="stRadio"] span {
                color: #FFFFFF !important;
            }
            
            /* All buttons purple hover */
            button:hover {
                border: 2px solid #B19CD9 !important;
            }
        </style>
    """, unsafe_allow_html=True)

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

    # Purple alert boxes ONLY for this tab
    st.markdown("""
    <style>
                
    div[data-testid="stAlert"] {
        background-color: #2b0a3d !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
    }

    div[data-testid="stAlert"] p {
        color: #FFFFFF !important;
    }
                
    /* Browse files hover text purple */
    [data-testid="stFileUploader"] button:hover {
    color: #B19CD9 !important;
    border: 2px solid #B19CD9 !important;
    }

    /* Make sure the text inside button changes too */
    [data-testid="stFileUploader"] button:hover * {
    color: #B19CD9 !important;
    }

    /* Download zip button - black background with white text */
    [data-testid="stDownloadButton"] button {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #000000 !important;
    }
    [data-testid="stDownloadButton"] button p {
        color: #FFFFFF !important;
    }
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

    # ── Start New Session ────────────────────────────────────────────────────
    if not st.session_state.known_data.empty or not st.session_state.unknown_data.empty:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("**Start New Session**", use_container_width=True):
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

    # ── STEP 1: Source folder path ───────────────────────────────────────────
    st.markdown("### Step 1 — Enter the source folder path")

    import os
    source_input = st.text_input(
        "Source folder path *",
        value=st.session_state.source_folder,
        placeholder=r"e.g.  /Users/you/BatRecordings  or  C:\BatRecordings",
        key="source_folder_input"
    )
    if st.button("**Verify Path & Load Files**", use_container_width=True):
        source_input = (source_input or "").strip()
        if not source_input:
            st.error("Please enter a folder path.")
        elif not os.path.isdir(source_input):
            st.error(f"Path not found or is not a folder: '{source_input}'")
        else:
            wav_files = [f for f in os.listdir(source_input)
                         if f.lower().endswith('.wav')]
            if not wav_files:
                st.warning(f"No .wav files found in '{source_input}'.")
            else:
                st.session_state.source_folder = source_input
                st.session_state.uploaded_files = wav_files
                # Reset results if the folder changed
                st.session_state.known_data = pd.DataFrame(columns=['Filename', 'Species Prediction', 'Confidence Level'])
                st.session_state.unknown_data = pd.DataFrame(columns=['Filename'])
                st.session_state.id_folder_result  = None
                st.session_state.unk_folder_result = None
                st.success(f"✅ Found {len(wav_files)} .wav file(s) in '{source_input}'.")
                st.rerun()

    # Show current verified folder and file count
    if st.session_state.source_folder and st.session_state.uploaded_files:
        st.info(f"**Source:** `{st.session_state.source_folder}` — "
                f"**{len(st.session_state.uploaded_files)}** .wav file(s) ready for classification.")

    st.markdown("---")

    # ── STEP 2: Classify ─────────────────────────────────────────────────────
    st.markdown("### Step 2 — Classify")

    classify_disabled = not bool(st.session_state.source_folder and st.session_state.uploaded_files)

    with st.form("classify_form"):
        submitted = st.form_submit_button(
            "Classify",
            use_container_width=True,
            disabled=classify_disabled
        )

        if submitted and not classify_disabled:
            with st.spinner("Analyzing audio files… Please wait."):
                time.sleep(2)
                known_df, unknown_df = process_audio_files(st.session_state.uploaded_files)

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
                st.markdown("The folder will be created if it does not exist, and files will be "
                            "**moved** from the source folder.")
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
                    id_folder_path = (id_folder_path or "").strip()
                    if not id_folder_path:
                        st.error("Please enter a destination folder path.")
                    elif os.path.abspath(id_folder_path) == os.path.abspath(st.session_state.source_folder):
                        st.error("Destination must be different from the source folder.")
                    else:
                        known_names = st.session_state.known_data['Filename'].tolist()
                        try:
                            os.makedirs(id_folder_path, exist_ok=True)
                        except Exception as exc:
                            st.error(f"Could not create folder: {exc}")
                            st.stop()
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
                    f"Moved **{len(res['moved'])}** file(s) to `{res['path']}`."
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
                    unk_folder_path = (unk_folder_path or "").strip()
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

            if not errors:
                duplicate = any(
                    s["Abbreviation"].lower() == abbr.lower() and
                    s["Latin Name"].lower() == latin.lower()
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

    submitted_training = st.button("Save Training Data", use_container_width=True, type="primary")

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

            st.session_state.training_entries.append({
                "Species": selected_species,
                "Detector": selected_detector,
                "FileCount": len(training_files),
                "FileNames": new_file_names
            })
            st.session_state.training_uploader_key += 1
            st.success(
                f"Training data saved: {len(training_files)} file(s) for {selected_species} at detector '{selected_detector}'.")
            st.rerun()

    st.markdown("---")

    # Show training data entries
    if st.session_state.training_entries:
        st.subheader("Training Data Entries")
        training_df = pd.DataFrame([{
            "Species": entry["Species"],
            "Detector": entry.get("Detector", entry.get("Location", "")),
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

# ============================================================================
# TAB 5: TRAIN NEW MODEL
# ============================================================================
with tab5:
    st.markdown("---")
    st.header("Train New Model")
    st.markdown("---")

    if not st.session_state.detectors:
        st.info("No detectors registered yet. Please add detectors in the 'Add Detector' tab first.")
    elif not st.session_state.species:
        st.info("No species registered yet. Please add species in the 'Add Species' tab first.")
    else:
        st.markdown("Select the detectors and species you want to include in the new model training run.")
        st.markdown("---")

        if "train_selections" not in st.session_state:
            st.session_state.train_selections = {}

        all_species_options = [s["Abbreviation"] for s in st.session_state.species]
        all_detector_ids = [d["Detector ID"] for d in st.session_state.detectors]

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

        for det_id in filtered_detectors:
            # Initialise selection for this detector if not present
            if det_id not in st.session_state.train_selections:
                st.session_state.train_selections[det_id] = {
                    "selected": False,
                    "species": [],
                    "species_search": ""
                }

            col_check, col_label = st.columns([0.05, 0.95])
            with col_check:
                selected = st.checkbox(
                    "",
                    value=st.session_state.train_selections[det_id]["selected"],
                    key=f"det_check_{det_id}"
                )
            with col_label:
                st.markdown(f"**{det_id}**")

            st.session_state.train_selections[det_id]["selected"] = selected

            if selected:
                st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;*Select species:*")
                chosen_species = []
                for sp in all_species_options:
                    already = sp in st.session_state.train_selections[det_id]["species"]
                    checked = st.checkbox(
                        sp,
                        value=already,
                        key=f"sp_check_{det_id}_{sp}"
                    )
                    if checked:
                        chosen_species.append(sp)
                st.session_state.train_selections[det_id]["species"] = chosen_species

            st.markdown("---")

        # Summary and train button
        active = {
            det: info for det, info in st.session_state.train_selections.items()
            if info["selected"]
        }

        if active:
            st.markdown("**Selected for training:**")
            all_valid = True
            for det_id, info in active.items():
                if info["species"]:
                    st.markdown(f"- **{det_id}**: {', '.join(info['species'])}")
                else:
                    st.markdown(f"- **{det_id}**: No species selected")
                    all_valid = False

            st.markdown("")
            if st.button("Train Model", use_container_width=True, type="primary"):
                if not all_valid:
                    st.error("Please select at least one species for each chosen detector before training.")
                else:
                    st.success(
                        f"Training job submitted for {len(active)} detector(s): "
                        + ", ".join(active.keys())
                    )
        else:
            st.info("Select at least one detector above to configure your training run.")

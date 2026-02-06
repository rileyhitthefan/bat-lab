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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def process_audio_files(uploaded_files):
    """Process uploaded audio files and classify them as KNOWN or UNKNOWN."""
    known_results = []
    unknown_results = []

    for file in uploaded_files:
        filename = file.name
        import random

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
tab1, tab2, tab3, tab4 = st.tabs(["Classify", "Add Detector", "Add Species", "Add Training Data"])

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
                
    </style>
    """, unsafe_allow_html=True)

    

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
                        time.sleep(2)
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

                    st.success(f"✅ Classification complete! Processed {len(uploaded)} file(s).")
                    st.rerun()

    st.markdown("---")

    # Display known species
    if not st.session_state.known_data.empty:
        st.subheader("✅ Identified Species")
        st.dataframe(
            st.session_state.known_data,
            use_container_width=True,
            hide_index=True,
            height=400
        )
    else:
        st.info("No identified species yet. Upload files to get started!")

    st.markdown("---")

    # Display unknown species
    if not st.session_state.unknown_data.empty:
        st.subheader("❓ Unknown Species")
        st.dataframe(
            st.session_state.unknown_data,
            use_container_width=True,
            hide_index=True,
            height=400
        )

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
        latin = st.text_input("Latin Name (optional)", placeholder="e.g., Myotis lucifugus")
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
                    st.session_state.training_entries.append({
                        "Species": selected_species,
                        "Location": selected_location,
                        "FileCount": len(training_files),
                        "FileNames": [f.name for f in training_files]
                    })
                    st.success(
                        f"Training data saved: {len(training_files)} file(s) for {selected_species} at {selected_location}")
                    st.rerun()

        if cancel_training:
            st.info("Cancelled - no changes made.")

    st.markdown("---")

    # Show training data entries
    if st.session_state.training_entries:
        st.subheader("Training Data Entries")
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

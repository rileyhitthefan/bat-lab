# connection.py
import sqlite3
import streamlit as st

def save_to_database():
    """
    Saves detector and species data from session state to the Batlab.sql database.
    """
    conn = None  # Initialize connection to None
    try:
        conn = sqlite3.connect('Batlab.sql')
        cursor = conn.cursor()

        # Save Detectors
        if st.session_state.detectors:
            detector_data = [(d['Detector'], d['Latitude'], d['Longitude']) for d in st.session_state.detectors]
            cursor.executemany('INSERT INTO locations (Area_name, Latitude, Longitude) VALUES (?, ?, ?)', detector_data)
            st.session_state.detectors = [] # Clear session state after saving

        # Save Species
        if st.session_state.species:
            species_data = [(s['Abbreviation'], s['LatinName'], s['CommonName']) for s in st.session_state.species]
            cursor.executemany('INSERT INTO bats (Abbreviation, LatinName, CommonName) VALUES (?, ?, ?)', species_data)
            st.session_state.species = [] # Clear session state after saving

        conn.commit()
        st.success("✅ Data saved to database.")

    except sqlite3.Error as e:
        st.error(f"❌ Database error: {e}")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
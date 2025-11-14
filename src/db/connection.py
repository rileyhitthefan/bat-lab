# connection.py
import sqlite3
import streamlit as st
import pandas as pd


# Sends training calls to DB from UI
def send_training_data():
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

        # Save Testing Data
        if st.session_state.training_entries:
            species_data = [(t['Species'], t['Location'], t['file']) for t in st.session_state.species]
            cursor.executemany('INSERT INTO bats (Species, Location, File) VALUES (?, ?, ?)', species_data)
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

# Pulls training calls from DB to run in ML model
def get_call_library_data(self, location=None):
    conn = None  # Initialize connection to None

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('batlab.SQL')

        # Define the SQL query
        query = "SELECT file, bat, location FROM call_library"
        if location:
                query += f" WHERE location = '{location}'"

        # Execute the query and load the results into a pandas DataFrame
        call_library_df = pd.read_sql_query(query, conn)

        # Display the first few rows of the DataFrame
        # display(call_library_df.head())

    except FileNotFoundError:
        print(f"Error: The database file was not found at {db_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the database connection
        if 'conn' in locals() and conn:
            conn.close()


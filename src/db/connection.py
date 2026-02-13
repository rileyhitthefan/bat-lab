# connection.py
import sqlite3
import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector import Error

config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'test', # Placeholder password for testing. In order to establish a connection, replace with new password
    'database': 'batlab_schema', # Placeholder database name for testing. In order to establish a connection, replace with new database name
    'auth_plugin': 'mysql_native_password'
}

# Sends training calls to DB from UI
def send_training_data(input_field):
    """
    Saves detector, species, and training data from session state to the Batlab.sql database.
    """
    conn = None  # Initialize connection to None
    try:
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()

        # Save Detectors
        if input_field == "detectors" and st.session_state.detectors:
            detector_data = [(d['Detector'], d['Latitude'], d['Longitude']) for d in st.session_state.detectors]
            cursor.executemany('INSERT INTO locations (Area_name, Latitude, Longitude) VALUES (%s, %s, %s)', detector_data)
            st.session_state.detectors = [] # Clear session state after saving

        # Save Species
        if input_field == "species" and st.session_state.species:
            species_data = [(s['Abbreviation'], s['LatinName'], s['CommonName']) for s in st.session_state.species]
            cursor.executemany('INSERT INTO bats (Abbreviation, LatinName, CommonName) VALUES (%s, %s, %s)', species_data)
            st.session_state.species = [] # Clear session state after saving

        # Save Training Data
        if input_field == "training_data" and st.session_state.training_entries:
            training_data = [(t['Species'], t['Location'], t['file']) for t in st.session_state.training_entries]
            cursor.executemany('INSERT INTO bats (Species, Location, File) VALUES (%s, %s, %s)', training_data)
            st.session_state.training_entries = [] # Clear session state after saving

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
        conn = mysql.connector.connect(**config)

        # Define the SQL query
        query = "SELECT file, bat, location FROM call_library"
        if location:
                query += f" WHERE location = '{location}'"

        # Execute the query and load the results into a pandas DataFrame
        call_library_df = pd.read_sql_query(query, conn)

        # Display the first few rows of the DataFrame
        # display(call_library_df.head())

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the database connection
        if conn:
            conn.close()


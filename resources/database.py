import os
import sqlite3
import pandas as pd
from tqdm import tqdm


def create_table(database_path, table_name):
    """Create a table in the database."""
    # Ensure the directory for the database exists
    ensure_directory_exists(database_path)
    
    # Create database connection
    try:
        with sqlite3.connect(database_path) as conn:
            curs = conn.cursor()
            create_table_statement = f"""CREATE TABLE IF NOT EXISTS {table_name} (
                                        imageid INTEGER PRIMARY KEY,
                                        filepath TEXT NOT NULL,
                                        filename TEXT NOT NULL,
                                        size TEXT NOT NULL)"""
            curs.execute(create_table_statement)
            conn.commit()
            print(f"Table '{table_name}' created successfully.")
    except sqlite3.Error as e:
        print(f"An error occurred while creating the table: {e}")


def insert_data_into_table(database_path, table_name, imageid, image_data):
    """Insert data into the table."""
    try:
        with sqlite3.connect(database_path) as conn:
            curs = conn.cursor()
            insert_statement = f"INSERT INTO {table_name} (imageid, filepath, filename, size) VALUES (?, ?, ?, ?)"
            curs.execute(insert_statement, (imageid, *image_data))
            conn.commit()
    except sqlite3.Error as e:
        print(f"An error occurred while inserting data: {e}")


def drop_table(database_path, table_name):
    """Drop the table from the database."""
    with sqlite3.connect(database_path) as conn:
        curs = conn.cursor()
        curs.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()


def ensure_directory_exists(database_path):
    """Ensure the directory for the given file path exists."""
    directory = os.path.dirname(database_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")


def get_image_path(database_path, table_name, image_id):
    """Get image path based on image id"""
    with sqlite3.connect(database_path) as conn:
        curs = conn.cursor()
        curs.execute(f"SELECT filepath, filename FROM {table_name} WHERE imageid = ?", (image_id,))
        result = curs.fetchone()  # Fetch the first matching row
        filepath, filename = result
        return os.path.join(filepath, filename)  # Combine the path and filename


def insert_data_from_pickle(pickle_file, database_path, table_name):
    # Load the pickle file into a pandas DataFrame
    images = pd.DataFrame(pd.read_pickle(pickle_file))

    # Insert DataFrame rows into the database
    for index, row in tqdm(images.iterrows(), total=len(images), desc="Inserting data into database"):
        imageid = row['image_id']
        filepath = row['root']
        filename = row['file']
        size = row.get('size', 'Unknown')

        insert_data_into_table(database_path, table_name, imageid, (filepath, filename, size))

pickle_file = 'image_info_T7_1.pkl'

# Define database path and table name
database_path = '/Volumes/T7 Shield 1/Uni/4. Semester/Big Data Engineering/image_database.db'
table_name = 'image_database_T7_1'

# Create the table if it doesn't exist
# create_table(database_path, table_name)

# insert data from pickle file into database
# insert_data_from_pickle(pickle_file, database_path, table_name)
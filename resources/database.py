import os
import sqlite3
from PIL import Image



def create_table(database_path, table_name):
    """Create a table in the database."""
    # Ensure the directory for the database exists
    ensure_directory_exists(database_path)
    
    # Create database connection
    with sqlite3.connect(database_path) as conn:
        curs = conn.cursor()
        create_table_statement = f"""CREATE TABLE IF NOT EXISTS {table_name} (
                                    imageid INTEGER PRIMARY KEY,
                                    filepath TEXT NOT NULL,
                                    filename TEXT NOT NULL,
                                    size TEXT NOT NULL)"""
        curs.execute(create_table_statement)
        conn.commit()


def insert_data_into_table(database_path, table_name, imageid, image_data):
    """Insert data into the table."""
    with sqlite3.connect(database_path) as conn:
        curs = conn.cursor()
        insert_statement = f"INSERT INTO {table_name} (imageid, filepath, filename, size) VALUES (?, ?, ?, ?)"
        curs.execute(insert_statement, (imageid, *image_data))
        conn.commit()


def get_picture_and_load_it(database_path, table_name):
    """Fetch and display the first image from the table."""
    with sqlite3.connect(database_path) as conn:
        curs = conn.cursor()
        curs.execute(f"SELECT filepath, filename FROM {table_name}")
        firstpic = curs.fetchall()[0]
        path = os.path.join(firstpic[0], firstpic[1])
        image = Image.open(path)
        image.show()


def drop_table(database_path, table_name):
    """Drop the table from the database."""
    with sqlite3.connect(database_path) as conn:
        curs = conn.cursor()
        curs.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()


def describe_table(database_path, table_name):
    """Describe the structure of the table."""
    with sqlite3.connect(database_path) as conn:
        curs = conn.cursor()
        curs.execute(f"PRAGMA table_info({table_name})")
        return curs.fetchall()


def select_all_images(database_path, table_name):
    """Select all images from the table."""
    with sqlite3.connect(database_path) as conn:
        curs = conn.cursor()
        curs.execute(f"SELECT * FROM {table_name}")
        return curs.fetchall()


def delete_all_images(database_path, table_name):
    """Delete all images from the table."""
    with sqlite3.connect(database_path) as conn:
        curs = conn.cursor()
        curs.execute(f"DELETE FROM {table_name}")
        conn.commit()


def ensure_directory_exists(database_path):
    """Ensure the directory for the given file path exists."""
    directory = os.path.dirname(database_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

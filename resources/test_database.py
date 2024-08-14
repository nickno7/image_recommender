import os
import sqlite3
from database import get_image_path, create_table, insert_data_into_table

def test_get_image_path(self):
        database_path = '/Volumes/T7 Shield 1/Uni/4. Semester/Big Data Engineering/image_database.db'
        table_name = 'image_database_T7_1'   
        image_id = 1

        # Test the get_image_path function
        result = get_image_path(database_path, table_name, image_id)
        expected_result = "/Volumes/T7 Shield 1/Downloads/ILSVRC/Data/CLS-LOC/all_images/n03982430_20160.JPEG"
        assert result == expected_result

def test_create_table():
    database_path = '/Volumes/T7 Shield 1/Uni/4. Semester/Big Data Engineering/test_database.db'
    table_name = 'test_table'

    # Ensure the database file does not exist before testing
    if os.path.exists(database_path):
        os.remove(database_path)

    # Create the table
    create_table(database_path, table_name)

    # Verify that the table was created successfully
    with sqlite3.connect(database_path) as conn:
        curs = conn.cursor()
        curs.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        result = curs.fetchone()

    assert result is not None, "Table creation failed"
    assert result[0] == table_name, "Table name does not match"

def test_insert_data_into_table():
    database_path = '/Volumes/T7 Shield 1/Uni/4. Semester/Big Data Engineering/test_database.db'
    table_name = 'test_table'
    imageid = 1
    image_data = ("/Users/nick/Downloads", "beach.jpeg", "9299")

    # Ensure the database file does not exist before testing
    if os.path.exists(database_path):
        os.remove(database_path)

    # Create the table before inserting data
    create_table(database_path, table_name)

    # Insert the data
    insert_data_into_table(database_path, table_name, imageid, image_data)

    # Verify that the data was inserted correctly
    with sqlite3.connect(database_path) as conn:
        curs = conn.cursor()
        curs.execute(f"SELECT * FROM {table_name} WHERE imageid = ?", (imageid,))
        result = curs.fetchone()

    assert result is not None, "Data insertion failed"
    assert result[0] == imageid, "Image ID does not match"
    assert result[1:] == image_data, "Image data does not match"

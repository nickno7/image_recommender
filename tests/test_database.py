import os
import sqlite3
from resources.database import get_image_path, create_table, insert_data_into_table
import tempfile

def create_temp_file():
    """Create a temporary database file and return the path."""
    # Create a temporary file for the database
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    db_path = temp_file.name
    temp_file.close()
    return db_path

def create_dummy_database():
    """Create a temporary database with a table and sample data."""
    db_path = create_temp_file()
    table_name = 'test_table'
    
    # Create the table
    create_table(db_path, table_name)
    
    # Insert sample data into the table
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f'''
        INSERT INTO {table_name} (imageid, filepath, filename, size) VALUES (?, ?, ?, ?)
        ''', (1, '/path/to/test', 'image.jpg', '133703'))
        conn.commit()
    
    return db_path, table_name

def test_get_image_path():
    db_path, table_name = create_dummy_database()
    
    image_id = 1
    expected_path = '/path/to/test/image.jpg'
    
    # Call the function
    result = get_image_path(db_path, table_name, image_id)
    
    # Assertions
    assert result == expected_path, f"Expected {expected_path} but got {result}"
    
    # Clean up
    os.remove(db_path)


def test_create_table():
    db_path = create_temp_file()
    table_name = 'test_table'
    
    # Create the table
    create_table(db_path, table_name)
    
    # Verify that the table was created successfully
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        result = cursor.fetchone()
    
    assert result is not None, "Table creation failed"
    assert result[0] == table_name, "Table name does not match"
    
    # Clean up
    os.remove(db_path)

def test_insert_data_into_table():
    db_path = create_temp_file()
    table_name = 'test_table'
    imageid = 1
    image_data = ("/Users/nick/Downloads", "beach.jpeg", "9299")

    # Create the table before inserting data
    create_table(db_path, table_name)

    # Insert the data
    insert_data_into_table(db_path, table_name, imageid, image_data)

    # Verify that the data was inserted correctly
    with sqlite3.connect(db_path) as conn:
        curs = conn.cursor()
        curs.execute(f"SELECT * FROM {table_name} WHERE imageid = ?", (imageid,))
        result = curs.fetchone()

    assert result is not None, "Data insertion failed"
    assert result[0] == imageid, "Image ID does not match"
    assert result[1:] == image_data, "Image data does not match"

    os.remove(db_path)

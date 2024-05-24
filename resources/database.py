import os
import sqlite3
from PIL import Image

def create_table(database_path, table_name):
    #set paths and create dir where database is stored",

    #create database connection
    conn = sqlite3.connect(database_path)
    curs = conn.cursor()
    
    create_table_statement = f"""CREATE TABLE if not exists {table_name}
                    (imageid INTEGER PRIMARY KEY,
                     filepath text not null,
                     filename text not null,
                     size text not null
                     )"""
    
    curs.execute(create_table_statement)
    
    conn.commit()
    conn.close()


def insert_data_into_table(database_path, table_name, imageid, image_data):
       
    conn = sqlite3.connect(database_path)
    curs = conn.cursor()

    insert_statement = f"INSERT INTO {table_name} (imageid, filepath, filename, size) VALUES (?, ?, ?, ?)"
    curs.execute(insert_statement, (imageid, *image_data))


def get_picture_and_load_it(database_path, table_name):
        conn = sqlite3.connect(database_path)
        curs = conn.cursor()

        curs.execute(f"SELECT filepath, filename FROM {table_name}")
        firstpic = curs.fetchall()[0]
        path = os.path.join(firstpic[0], firstpic[1])
        image = Image.open(path)
        image.show()



def drop_table(database_path, table_name):
        conn = sqlite3.connect(database_path)
        curs = conn.cursor()

        for name in table_name:
               curs.execute(f"DROP TABLE if exists {name}")
        
        conn.commit()
        conn.close()

def describe_table(database_path, table_name):
        conn = sqlite3.connect(database_path)
        curs = conn.cursor()

        # check if table is created
        #print(f"Database created: {os.listdir(database_dir_path)}")

        curs.execute(f"PRAGMA table_info({table_name})")
        
        return curs.fetchall()



def select_all_images(database_path, table_name):
        conn = sqlite3.connect(database_path)
        curs = conn.cursor()

        curs.execute(f"SELECT * FROM {table_name}")
        return curs.fetchall()


def delete_all_images(database_path, table_name):
        conn = sqlite3.connect(database_path)
        curs = conn.cursor()

        curs.execute(f"DELETE FROM {table_name}")

import os

#base dir where script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#database directory and path
DATABASE_DIR = os.path.join(BASE_DIR, 'databases')
DATABASE_NAME = 'image_recommender.db'
DATABASE_PATH = os.path.join(DATABASE_DIR, DATABASE_NAME)

# path to harddrive images
IMAGEPATH_HARDDRIVE = 'F:\\data\\image_data'

#tablename
TABLE_NAME = "images"
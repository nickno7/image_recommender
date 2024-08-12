import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle
import time
import sqlite3


def get_vector(image_path, bins=32):
    # check whether the file exists
    if not os.path.isfile(image_path):
        raise ValueError(f"Die Datei {image_path} existiert nicht.")
    
    # load image
    image = cv2.imread(image_path)
    
    # checker whether the image has been loaded successfully
    if image is None:
        raise ValueError(f"Das Bild {image_path} konnte nicht geladen werden.")

    # extract rgb information of the image
    red = cv2.calcHist([image], [2], None, [bins], [0, 256])
    green = cv2.calcHist([image], [1], None, [bins], [0, 256])
    blue = cv2.calcHist([image], [0], None, [bins], [0, 256])
    
    vector = np.concatenate([red, green, blue], axis=0).reshape(-1)
    return vector

# function to save the embeddings in a pickle file
def save_progress(vectors, filename):
    with open(filename, 'wb') as f:
        pickle.dump(vectors, f)

# function to load existing pickle files and continue with them (in case the process crashed)
def load_progress(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return {}

# Load image metadata from the database
def load_image_database(database_path, table_name):
    with sqlite3.connect(database_path) as conn:
        curs = conn.cursor()
        curs.execute(f"SELECT imageid, filepath, filename FROM {table_name}")
        return curs.fetchall()

def main():
    saveFile = 'color_vectors.pkl'

    # store all color vector
    color_vectors = load_progress(saveFile)

    # check whether the pickle file already contains embeddings of images in the input directory
    # to prevent from loading again when they already exist
    processed_images = set(color_vectors.keys())

    # Load image metadata
    images = pd.DataFrame(pd.read_pickle("image_info.pkl"))

    print("Processing Images...")
    # calculating the color embeddings for every image in the database
    for index, row in tqdm(images.iterrows(), total=len(images), desc="Processing images"):
        image_id = row['image_id']

        # skip already processed images
        if image_id in processed_images:
            continue

        image_path = os.path.join(row['root'], row['file'])
        try:
            # calculate color embedding
            vector = get_vector(image_path)
            if vector is not None:
                color_vectors[image_id] = vector
            # save progress every 5000th image
            if len(color_vectors) % 50000 == 0:
                save_progress(color_vectors, saveFile)
        # prevent the code from breaking
        except ValueError as e:
            print(f"Error opening/processing image {image_id}: {e}")


    # save final progress
    save_progress(color_vectors, saveFile)
    print(f"Number of vectors generated: {len(color_vectors)}")

if __name__ == "__main__":
    main()
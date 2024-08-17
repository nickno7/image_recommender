from skimage.io import imread, imshow
from skimage.transform import resize 
from skimage.feature import hog
from skimage.color import rgb2gray
import pandas as pd
import pickle
import os
from tqdm import tqdm
import numpy as np


# Function to extract HOG features from an image
def extract_hog_features(image_path):
    try:
        img = imread(image_path)

        # Ensure the image has 3 channels
        if img.ndim == 2:  # Grayscale image
            img = np.stack((img,) * 3, axis=-1)
        elif img.shape[2] == 4:  # If the image has an alpha channel (RGBA), drop it
            img = img[..., :3]

        resized_img = resize(img, (128, 64))
        gray_img = rgb2gray(resized_img)
        features = hog(gray_img, orientations=6, pixels_per_cell=(16, 16),
                          cells_per_block=(2, 2), visualize=False)
        
        return features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

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

def main():

    # pickle file with the hog embeddings
    saveFile = "hog_vectors.pkl"
    # load previous progress, when existing
    allVectors = load_progress(saveFile)

    # check whether the pickle file already contains embeddings of images in the input directory
    # to prevent from loading again when they already exist
    processed_images = set(allVectors.keys())

    # Load image metadata
    images = pd.DataFrame(pd.read_pickle("image_info_T7_1.pkl"))

    # calculating the image embeddings for every image in the database
    print("Converting images to feature vectors:")
    for index, row in tqdm(images.iterrows(), total=len(images), desc="Processing images"):
        image_id = row['image_id']

        # skip already processed images
        if image_id in processed_images:
            continue

        image_path = os.path.join(row['root'], row['file'])
        try:
            features = extract_hog_features(image_path)
            
            if features is not None:
                allVectors[image_id] = features

            # save the progress every 50000th image
            if len(allVectors) % 50000 == 0:
                save_progress(allVectors, saveFile)
        # prevent the code from breaking
        except Exception as e:
            print(f"Error opening/processing image {image_id}: {e}")

    # save everything in the pickle file 
    save_progress(allVectors, saveFile)
    print(f"Number of vectors generated: {len(allVectors)}")

if __name__ == "__main__":
    main()

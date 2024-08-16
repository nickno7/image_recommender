import cv2
import numpy as np
from resources.color_embeddings import get_vector
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from resources.database import get_image_path
from resources.autoencoder_resnet18 import Img2VecResnet18
from PIL import Image
import joblib
# from resources.hog_embeddings import extract_hog_features

# function to display an image
def show_image(image_path, title):
    """Display an image with a title."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def load_embeddings(embeddings_file):
    """Load and process embedding data into separate lists."""
    image_ids, vectors = zip(*embeddings_file.items())
    embeddings = np.array(vectors)
    return image_ids, embeddings

def calculate_scores(query_vector, embeddings, image_ids, number_pictures):
    # calculate similarity scores
    scores = cosine_similarity([query_vector], embeddings)[0]
    # Find the closest 10 vectors and their similarity scores
    closest_indices = np.argsort(-scores)[:number_pictures]
    closest_vectors = scores[closest_indices]
    
    return closest_vectors, closest_indices

def process_input_image(image_path, mode, img2vec, ipca):
    """Process the input image to generate an embedding vector based on the selected mode."""
    if mode == "color":
        return get_vector(image_path)
    elif mode == "content":
        img = Image.open(image_path)
        return img2vec.getVec(img)
    # elif mode == "pca":
    #     vector = preprocess_image(image_path)
    #     reduced_vector = ipca.transform([vector])
    #     return reduced_vector.flatten()
    # elif mode == "hog":
    #     vector = extract_hog_features(image_path)
    #     return vector

    else:
        raise ValueError("Invalid mode. Choose either 'color', 'content' or 'pca'.")

def get_similar_images(image_path, database_path, table_name, mode, embeddings_file, number_pictures, several_inputs):
    img2vec = Img2VecResnet18()   
    # Load the IncrementalPCA model
    ipca = joblib.load('pca_model.pkl')    

    # get similar images for more than one input image
    if several_inputs:
        query_vectors = []
        for image in image_path:
            query_vector = process_input_image(image, mode, img2vec, ipca)
            query_vectors.append(query_vector)
        # Calculate the average embedding across all input images
        query_vector = np.mean(query_vectors, axis=0)
    else:
        query_vector = process_input_image(image_path, mode, img2vec, ipca)

    image_ids, embeddings = load_embeddings(embeddings_file)

    closest_vectors, closest_indices = calculate_scores(query_vector, embeddings, image_ids, number_pictures)           

    # Display input image(s)
    if several_inputs:
        for image in image_path:
            show_image(image, "Input Image")
    else:
        show_image(image_path, "Input Image")


    # Prepare paths and titles for similar images
    closest_image_paths = [get_image_path(database_path, table_name, image_ids[i]) for i in closest_indices]
    titles = [f"ID: {image_ids[i]}, Score: {closest_vectors[j]:.2f}" for j, i in enumerate(closest_indices)]

    for image_path, title in zip(closest_image_paths, titles):
        show_image(image_path, title)


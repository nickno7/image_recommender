import cv2
import numpy as np
from color_embeddings import get_vector
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from database import get_image_path
from autoencoder_resnet18 import Img2VecResnet18
from PIL import Image


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

def process_input_image(image_path, mode, img2vec):
    """Process the input image to generate an embedding vector based on the selected mode."""
    if mode == "color":
        return get_vector(image_path)
    elif mode == "content":
        I = Image.open(image_path)
        return img2vec.getVec(I)
    else:
        raise ValueError("Invalid mode. Choose either 'color' or 'content'.")

def get_similar_images(image_path, database_path, table_name, mode, embeddings_file, number_pictures):
    img2vec = Img2VecResnet18()    
    
    query_vector = process_input_image(image_path, mode, img2vec)

    image_ids, embeddings = load_embeddings(embeddings_file)
    
    closest_vectors, closest_indices = calculate_scores(query_vector, embeddings, image_ids, number_pictures)

    # display input image
    show_image(image_path, "Input Image")

    # Prepare paths and titles for similar images
    closest_image_paths = [get_image_path(database_path, table_name, image_ids[i]) for i in closest_indices]
    titles = [f"ID: {image_ids[i]}, Score: {closest_vectors[j]:.2f}" for j, i in enumerate(closest_indices)]

    for image_path, title in zip(closest_image_paths, titles):
        show_image(image_path, title)


from .similarity_functions import load_embeddings, calculate_scores, process_input_image, get_similar_images
from .resnet_embeddings import Img2VecResnet18
import numpy as np
import pandas as pd
from PIL import Image
from unittest.mock import patch, Mock


def test_load_embeddings():
    embeddings_file = {
        1: [0.1, 0.2, 0.3],
        2: [0.4, 0.5, 0.6],
        3: [0.7, 0.8, 0.9],
    }
    image_ids, embeddings = load_embeddings(embeddings_file)
    
    assert image_ids == (1, 2, 3)
    np.testing.assert_array_equal(embeddings, np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]))

def test_calculate_scores():
    query_vector = np.array([0.5, 0.5, 0.5])
    embeddings = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])
    image_ids = [1, 2, 3]
    number_pictures = 2

    closest_vectors, closest_indices = calculate_scores(query_vector, embeddings, image_ids, number_pictures)
    
    assert len(closest_vectors) == len(closest_indices) == number_pictures
    assert closest_indices[0] == 2  # The most similar vector should be at index 2
    
def test_process_input_image_color():
    image_path = "/test_images/volleyball.jpeg"
    mode = "color"
    
    vector = process_input_image(image_path, mode, None)
    
    assert isinstance(vector, np.ndarray), "Output is not a numpy array"

def test_process_input_image_content():
    image_path = "/test_images/yoga.jpeg" 
    mode = "content"
    img2vec = Img2VecResnet18()

    vector = process_input_image(image_path, mode, img2vec)

    assert isinstance(vector, np.ndarray), "Output is not a numpy array"

def test_process_input_image_hog():
    image_path = "/test_images/beach_2.jpeg"
    mode = "hog"
    
    vector = process_input_image(image_path, mode, None)
    
    assert isinstance(vector, np.ndarray), "Output is not a numpy array"
    
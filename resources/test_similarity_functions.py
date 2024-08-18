from .similarity_functions import load_embeddings, calculate_scores, process_input_image, get_similar_images
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
    
def test_process_input_image():
    image_path = 'dummy_path.jpg'  # Use a dummy path

    # Test for 'color' mode
    mode_color = 'color'
    img2vec = None  # Not used in color mode

    # Mock the get_vector function
    with patch('similarity_functions.color_embeddings.get_vector', return_value=np.array([1, 2, 3])):
        result_color = process_input_image(image_path, mode_color, img2vec)
        expected_result_color = np.array([1, 2, 3])
        np.testing.assert_array_equal(result_color, expected_result_color)

    # Test for 'content' mode
    mode_content = 'content'
    img2vec = Mock()  # Create a mock Img2VecResnet18 instance
    img2vec.getVec.return_value = np.array([0.1, 0.2, 0.3, 0.4])  # Dummy embedding data

    # Mock the Image.open function to return a simple image
    with patch('similarity_functions.Image.open', return_value=Image.new('RGB', (10, 10))):
        result_content = process_input_image(image_path, mode_content, img2vec)
        expected_result_content = np.array([0.1, 0.2, 0.3, 0.4])
        np.testing.assert_array_equal(result_content, expected_result_content)

    # Test for 'hog' mode
    mode_hog = 'hog'
    # Mock the extract_hog_features function
    with patch('similarity_functions.hog_embeddings.extract_hog_features', return_value=np.array([0.5, 0.6, 0.7])):
        result_hog = process_input_image(image_path, mode_hog, img2vec)
        expected_result_hog = np.array([0.5, 0.6, 0.7])
        np.testing.assert_array_equal(result_hog, expected_result_hog)
    

def test_get_similar_images():
    # Mock input parameters
    image_path = 'dummy_path.jpg'
    database_path = 'dummy_database.db'
    table_name = 'dummy_table'
    embeddings_file = {
        1: np.array([0.1, 0.2, 0.3, 0.4]), 
        2: np.array([0.5, 0.6, 0.7, 0.8]),
        3: np.array([0.9, 1.0, 1.1, 1.2])
    }
    number_pictures = 2
    several_inputs = False

    # Mock Img2VecResnet18 and joblib.load
    mock_img2vec = Mock()
    mock_img2vec.getVec.return_value = np.array([0.1, 0.2, 0.3, 0.4])

    mock_get_image_path = lambda db, tbl, id: f"image_{id}.jpg"

    with patch('similarity_functions.resnet_embeddings.Img2VecResnet18', return_value=mock_img2vec), \
         patch('similarity_functions.database.get_image_path', side_effect=mock_get_image_path), \
         patch('similarity_functions.show_image'):

        # Test for 'color' mode
        with patch('similarity_functions.get_vector', return_value=np.array([0.1, 0.2, 0.3, 0.4])):
            get_similar_images(image_path, database_path, table_name, 'color', embeddings_file, number_pictures, several_inputs)

        # Test for 'content' mode
        with patch('similarity_functions.Image.open', return_value=Mock()), \
             patch.object(mock_img2vec, 'getVec', return_value=np.array([0.1, 0.2, 0.3, 0.4])):
            get_similar_images(image_path, database_path, table_name, 'content', embeddings_file, number_pictures, several_inputs)
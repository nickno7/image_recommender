# Image Recommender

This repository contains an image recommendation system that utilizes a dataset of half a million pictures. The algorithm accepts an input picture and searches for the most similar images in the dataset. The definition of similarity can vary based on the similarity measurement used, including color, content or motif.


## Features

- *Data Processing*: Load and preprocess image data for model training and recommendations.
- *Model Training*: Train a recommendation model using PyTorch.
- *Image Recommendation*: Recommend images based on similarity in color or content.
## Installation

1. Clone this repository:

    bash
    git clone https://github.com/nickno7/image_recommender.git
    

2. Navigate to the base directory:

    bash
    cd image_recommender
    

3. Install the required dependencies:

    bash
    pip install -r requirements.txt
    

## Usage 

You can adapt and reuse this repository for your own purposes by following these steps:

1. Navigate to the resources directory.

2. Modify the image path in pickle_generator.ipynb to point to your image dataset.

3. Run the cells in pickle_generator.ipynb to generate a pickle file containing image data.

4. Choose and run the appropriate similarity script to create a pickle file with embeddings:

   - For color similarity, run color_embeddings.py.
   - For motif similarity, run autoencoder_resnet18.py.
   - For other types of similarity, run the corresponding script.

   Note: These scripts may take some time to process the images.

5. Insert the data into the database:

    - adjust the variables pickle_file, database_path and table_name (pickle_file should be the filename you set in the pickle_generator.ipynb)
    - then run the commands create_table(database_path, table_name) & insert_data_from_pickle(pickle_file, database_path, table_name)

6. Use the similarity.ipynb Jupyter notebook to perform image recommendations. Set the following parameters:
    
    - database_path and table_name you created in step 6
    - image_path: input image for which you want to get similar images
      

    This will generate the top 5 similar images based on the selected mode(based on the cell you run). The modes are *content*, *color* and *hog*

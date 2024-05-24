import os 
from PIL import Image

#get resolution
def get_image_size(image_path):
    with Image.open(image_path) as img:
        size= img.size
    return str(size)

#generator
def load_images_from_harddrive(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                size = get_image_size(os.path.join(root, filename))
                yield root, filename, size

def get_image_count():
    pass
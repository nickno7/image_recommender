import os
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm
from torchvision import models
import pickle
import pandas as pd


# we use the resnet18 cnn model to obtain feature vectors
class Img2VecResnet18():
    def __init__(self):
        self.device = torch.device("cpu")
        self.numberFeatures = 512
        self.modelName = "resnet-18"
        self.model, self.featureLayer = self.getFeatureLayer()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize transformation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def getFeatureLayer(self):
        cnnModel = models.resnet18(pretrained=True)
        layer = cnnModel._modules.get('avgpool')
        self.layer_output_size = 512
        
        return cnnModel, layer
    
    def getVec(self, img):
        try:
            # to handle png images with transparent background
            if img.mode == 'RGBA':
                # Create a white background image
                white_bg = Image.new("RGB", img.size, (255, 255, 255))
                img = Image.alpha_composite(white_bg, img.convert('RGBA')).convert('RGB')
            # convert non rgb images
            if img.mode != 'RGB':
                img = img.convert('RGB')

            image = self.transform(img).unsqueeze(0).to(self.device)
            embedding = torch.zeros(1, self.numberFeatures, 1, 1)
            def copyData(m, i, o): embedding.copy_(o.data)
            h = self.featureLayer.register_forward_hook(copyData)
            self.model(image)
            h.remove()
            return embedding.numpy()[0, :, 0, 0]
        except Exception as e:
            print(f"Error processing image: {e}")
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
    # pickle file with the image embeddings
    saveFile = "image_vectors.pkl"
    # load previous progress, when existing
    allVectors = load_progress(saveFile)

    # check whether the pickle file already contains embeddings of images in the input directory
    # to prevent from loading again when they already exist
    processed_images = set(allVectors.keys())

    # Load image metadata
    images = pd.DataFrame(pd.read_pickle("image_info.pkl"))

    img2vec = Img2VecResnet18()

    # calculating the image embeddings for every image in the database
    print("Converting images to feature vectors:")
    for index, row in tqdm(images.iterrows(), total=len(images), desc="Processing images"):
        image_id = row['image_id']

        # skip already processed images
        if image_id in processed_images:
            continue

        image_path = os.path.join(row['root'], row['file'])
        try:
            img = Image.open(image_path)
            # calculate vector/embedding
            vec = img2vec.getVec(img)
            if vec is not None:
                allVectors[image_id] = vec
            img.close()
            # save the progress every 1000th image
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
    
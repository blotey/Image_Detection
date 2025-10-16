import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def load_dataset(data_folder, test_size=0.2):
    images = load_images_from_folder(data_folder)
    labels = [os.path.basename(data_folder)] * len(images)  # Assuming folder name is the label
    return train_test_split(images, labels, test_size=test_size, random_state=42)

def preprocess_images(images):
    processed_images = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = cv2.resize(img, (224, 224))  # Resize to 224x224
        processed_images.append(img)
    return np.array(processed_images)
import cv2
import numpy as np
import os
from src.datasets.loader import load_dataset
from src.recognizer.embeddings import EmbeddingsExtractor
from src.recognizer.classifier import FaceRecognizer

def train_model():
    # Load dataset
    images, labels = load_dataset()

    # Extract embeddings
    extractor = EmbeddingsExtractor()
    embeddings = extractor.extract_embeddings(images)

    # Train the face recognizer
    recognizer = FaceRecognizer()
    recognizer.train_model(embeddings, labels)

    # Save the trained model
    model_path = os.path.join('models', 'face_recognizer_model.h5')
    recognizer.save_model(model_path)
    print(f'Model saved to {model_path}')

if __name__ == "__main__":
    train_model()
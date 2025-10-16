import argparse
import cv2
import numpy as np
from src.detector.face_detector import FaceDetector
from src.recognizer.classifier import FaceRecognizer
from src.datasets.loader import load_test_dataset

def evaluate_model(detector, recognizer, test_dataset):
    correct_predictions = 0
    total_predictions = 0

    for image, label in test_dataset:
        faces = detector.detect_faces(image)
        for face in faces:
            embedding = recognizer.extract_embeddings(face)
            predicted_label = recognizer.recognize_face(embedding)
            if predicted_label == label:
                correct_predictions += 1
            total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f'Accuracy: {accuracy * 100:.2f}%')

def main(args):
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    
    test_dataset = load_test_dataset(args.dataset_path)
    evaluate_model(detector, recognizer, test_dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the face detection and recognition model.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the test dataset')
    args = parser.parse_args()
    main(args)
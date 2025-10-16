import cv2
import numpy as np
from detector.face_detector import FaceDetector
from recognizer.embeddings import EmbeddingsExtractor
from recognizer.classifier import FaceRecognizer
from utils.video import capture_video

def main():
    # Initialize face detector and recognizer
    face_detector = FaceDetector()
    embeddings_extractor = EmbeddingsExtractor()
    face_recognizer = FaceRecognizer()

    # Capture video from webcam
    capture_video(face_detector, embeddings_extractor, face_recognizer)

if __name__ == "__main__":
    main()
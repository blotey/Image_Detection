import cv2
import numpy as np
from src.detector.face_detector import FaceDetector
from src.recognizer.embeddings import EmbeddingsExtractor
from src.recognizer.classifier import FaceRecognizer

def main(video_source=0):
    face_detector = FaceDetector()
    embeddings_extractor = EmbeddingsExtractor()
    face_recognizer = FaceRecognizer()

    # Load the trained model for face recognition
    face_recognizer.load_model('path/to/trained_model')

    # Start video capture
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        faces = face_detector.detect_faces(frame)

        # Process each detected face
        for face in faces:
            # Extract embeddings
            embedding = embeddings_extractor.extract_embeddings(face)

            # Recognize the face
            label = face_recognizer.recognize_face(embedding)

            # Draw bounding box and label on the frame
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face Detection and Recognition', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
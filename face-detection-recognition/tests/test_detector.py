import unittest
import cv2
from src.detector.face_detector import FaceDetector

class TestFaceDetector(unittest.TestCase):

    def setUp(self):
        self.detector = FaceDetector()

    def test_detect_faces(self):
        image = cv2.imread('tests/test_image.jpg')  # Replace with a valid test image path
        faces = self.detector.detect_faces(image)
        self.assertIsInstance(faces, list)
        self.assertGreater(len(faces), 0, "No faces detected in the test image.")

    def test_draw_bounding_boxes(self):
        image = cv2.imread('tests/test_image.jpg')  # Replace with a valid test image path
        faces = self.detector.detect_faces(image)
        output_image = self.detector.draw_bounding_boxes(image, faces)
        self.assertIsNotNone(output_image)
        self.assertEqual(image.shape, output_image.shape, "Output image shape does not match input image shape.")

if __name__ == '__main__':
    unittest.main()
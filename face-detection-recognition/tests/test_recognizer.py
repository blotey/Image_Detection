import unittest
from src.recognizer.classifier import FaceRecognizer
from src.recognizer.embeddings import EmbeddingsExtractor

class TestFaceRecognizer(unittest.TestCase):

    def setUp(self):
        self.recognizer = FaceRecognizer()
        self.extractor = EmbeddingsExtractor()
        self.test_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        self.test_labels = ['person1', 'person2']

    def test_train_model(self):
        self.recognizer.train_model(self.test_embeddings, self.test_labels)
        self.assertTrue(hasattr(self.recognizer, 'model'))

    def test_recognize_face(self):
        self.recognizer.train_model(self.test_embeddings, self.test_labels)
        result = self.recognizer.recognize_face([0.1, 0.2, 0.3])
        self.assertEqual(result, 'person1')

    def test_recognize_unknown_face(self):
        self.recognizer.train_model(self.test_embeddings, self.test_labels)
        result = self.recognizer.recognize_face([0.7, 0.8, 0.9])
        self.assertEqual(result, 'Unknown')

if __name__ == '__main__':
    unittest.main()
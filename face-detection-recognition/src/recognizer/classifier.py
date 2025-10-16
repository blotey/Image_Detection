class FaceRecognizer:
    def __init__(self):
        self.model = None
        self.labels = None

    def train_model(self, embeddings, labels):
        from sklearn.svm import SVC
        from sklearn.preprocessing import LabelEncoder
        from sklearn.pipeline import make_pipeline

        self.labels = LabelEncoder().fit(labels)
        self.model = make_pipeline(SVC(kernel='linear', probability=True))
        self.model.fit(embeddings, self.labels.transform(labels))

    def recognize_face(self, embedding):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        prediction = self.model.predict([embedding])
        return self.labels.inverse_transform(prediction)[0]
# Configuration settings for the Face Detection and Recognition project

# Paths
DATASET_PATH = "path/to/dataset"
MODEL_SAVE_PATH = "path/to/save/model"
EMBEDDINGS_PATH = "path/to/save/embeddings"

# Model parameters
HAAR_CASCADE_PATH = "path/to/haarcascade_frontalface_default.xml"
CONFIDENCE_THRESHOLD = 0.5

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Other constants
IMAGE_SIZE = (224, 224)  # Resize images to this size for the model
NUM_CLASSES = 10  # Adjust based on the number of classes in the dataset
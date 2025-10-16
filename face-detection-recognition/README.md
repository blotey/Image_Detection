# Face Detection and Recognition System

This project implements a Face Detection and Recognition system using machine learning techniques. It leverages OpenCV and NumPy to detect and recognize human faces from images and video streams.

## Project Structure

```
face-detection-recognition
├── src
│   ├── main.py               # Entry point of the application
│   ├── config.py             # Configuration settings
│   ├── detector               # Face detection module
│   │   ├── __init__.py
│   │   ├── face_detector.py   # Face detection algorithms
│   │   └── utils.py          # Utility functions for detection
│   ├── recognizer             # Face recognition module
│   │   ├── __init__.py
│   │   ├── embeddings.py      # Facial embeddings extraction
│   │   └── classifier.py      # Face recognition logic
│   ├── datasets               # Dataset handling
│   │   ├── __init__.py
│   │   └── loader.py         # Dataset loading and preprocessing
│   └── utils                  # Utility functions
│       ├── video.py          # Video stream handling
│       └── preprocessing.py   # Image preprocessing functions
├── notebooks                  # Jupyter Notebooks for exploration and training
│   ├── 01-data-exploration.ipynb
│   └── 02-training-and-evaluation.ipynb
├── scripts                    # Scripts for training, evaluation, and inference
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
├── models                     # Model documentation
│   └── README.md
├── tests                      # Unit tests for the project
│   ├── test_detector.py
│   └── test_recognizer.py
├── .devcontainer              # Development container configuration
│   └── devcontainer.json
├── Dockerfile                 # Docker image definition
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment specification
├── .gitignore                 # Git ignore file
├── README.md                  # Project documentation
└── LICENSE                    # Licensing information
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd face-detection-recognition
pip install -r requirements.txt
```

Alternatively, you can create a conda environment using the `environment.yml` file:

```bash
conda env create -f environment.yml
```

## Usage

To run the face detection and recognition system, execute the following command:

```bash
python src/main.py
```

You can also explore the provided Jupyter Notebooks for data exploration and model training.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
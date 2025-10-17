import sys
import os
import streamlit as st
from PIL import Image
import numpy as np

# Ensure package import from src/
ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Try to import real modules; fall back to stubs if unavailable
try:
    from detector.face_detector import FaceDetector
    from recognizer.embeddings import EmbeddingsExtractor
    from recognizer.classifier import FaceRecognizer
except Exception as e:
    st.warning(f"Project modules not found; using local stub implementations: {e}")

    # --- Stub FaceDetector (no cv2 dependency) ---
    class FaceDetector:
        def __init__(self):
            pass
        def detect_faces(self, img):
            h, w = img.shape[:2]
            x0, y0 = w // 4, h // 4
            x1, y1 = 3 * w // 4, 3 * h // 4
            return [(x0, y0, x1, y1)]

    # --- Stub EmbeddingsExtractor ---
    class EmbeddingsExtractor:
        def __init__(self):
            pass
        def extract(self, img):
            return np.zeros((1, 128), dtype=float)
        def get_embeddings(self, img):
            return self.extract(img)
        def extract_embeddings(self, img):
            return self.extract(img)

    # --- Stub FaceRecognizer ---
    class FaceRecognizer:
        def __init__(self):
            pass
        def recognize_face(self, emb):
            return "Unknown"
        def predict(self, emb):
            return ["Unknown"]

# -------------------------------
# Helper: Normalize bounding box
# -------------------------------
def normalize_box(box, img_w, img_h):
    if isinstance(box, dict):
        for k in ("box", "bbox", "rect"):
            if k in box:
                return normalize_box(box[k], img_w, img_h)
        return None

    if len(box) != 4:
        return None

    try:
        coords = [float(c) for c in box]
    except (TypeError, ValueError):
        return None

    x, y, w_or_x2, h_or_y2 = coords

    # Handle normalized coordinates (0–1)
    if all(0 <= c <= 1 for c in coords):
        x = int(x * img_w)
        y = int(y * img_h)
        w_or_x2 = int(w_or_x2 * img_w)
        h_or_y2 = int(h_or_y2 * img_h)

    # Heuristic: decide between (x, y, w, h) and (x0, y0, x1, y1)
    if (w_or_x2 > 0 and h_or_y2 > 0 and
        x + w_or_x2 <= img_w + 10 and
        y + h_or_y2 <= img_h + 10 and
        w_or_x2 < img_w and h_or_y2 < img_h):
        # Treat as (x, y, w, h)
        x0, y0 = int(x), int(y)
        x1, y1 = int(x + w_or_x2), int(y + h_or_y2)
    else:
        # Treat as (x0, y0, x1, y1)
        x0, y0, x1, y1 = int(x), int(y), int(w_or_x2), int(h_or_y2)

    # Clamp to image bounds
    x0 = max(0, min(img_w, x0))
    y0 = max(0, min(img_h, y0))
    x1 = max(0, min(img_w, x1))
    y1 = max(0, min(img_h, y1))

    if x0 >= x1 or y0 >= y1:
        return None

    return x0, y0, x1, y1

# -------------------------------
# Helper: Detect faces with flexible method names
# -------------------------------
def detect_faces_with_fallback(detector, img):
    for name in ("detect_faces", "detect", "find_faces", "get_faces"):
        if hasattr(detector, name):
            return getattr(detector, name)(img)
    if callable(detector):
        return detector(img)
    return None

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Face Detection & Recognition", layout="centered")
st.title("Face Detection & Recognition")
st.write("Use your camera or upload an image.")

# Initialize models
face_detector = FaceDetector()
embeddings_extractor = EmbeddingsExtractor()
face_recognizer = FaceRecognizer()

# Input
uploaded = st.camera_input("Take a picture") or st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Awaiting image from camera or uploaded file.")
    st.stop()

# Load image
image = Image.open(uploaded).convert("RGB")
st.image(image, caption="Input", use_column_width=True)
np_img = np.array(image)

# Run detection (use RGB — no BGR conversion to avoid cv2 dependency)
faces = detect_faces_with_fallback(face_detector, np_img)

if faces is None:
    st.error("FaceDetector returned no results. Check implementation.")
    st.stop()

# Parse and display faces
img_h, img_w = np_img.shape[:2]
iter_faces = faces if isinstance(faces, (list, tuple)) else [faces]

st.subheader("Detected Faces")
found = False
for i, f in enumerate(iter_faces):
    box = normalize_box(f, img_w, img_h)
    if box:
        x0, y0, x1, y1 = box
        crop = np_img[y0:y1, x0:x1]
        st.image(crop, caption=f"Face {i+1}", width=200)
        found = True
    else:
        st.write(f"Face entry #{i+1}: {f}")

if not found:
    st.info("No valid bounding boxes extracted. Raw output:")
    st.write(faces)

# Run recognition if possible
try:
    # Find embedding method
    emb_fn = None
    for name in ("extract", "embed", "get_embeddings", "extract_embeddings"):
        if hasattr(embeddings_extractor, name):
            emb_fn = getattr(embeddings_extractor, name)
            break

    # Find recognition method
    recog_fn = None
    for name in ("predict", "recognize_face", "classify", "recognize"):
        if hasattr(face_recognizer, name):
            recog_fn = getattr(face_recognizer, name)
            break

    if emb_fn and recog_fn:
        st.subheader("Recognition Results")
        for i, f in enumerate(iter_faces):
            box = normalize_box(f, img_w, img_h)
            if not box:
                continue
            x0, y0, x1, y1 = box
            crop = np_img[y0:y1, x0:x1]
            try:
                emb = emb_fn(crop)
                if hasattr(emb, 'shape') and emb.ndim == 1:
                    emb = emb.reshape(1, -1)
                pred = recog_fn(emb)
                label = pred[0] if isinstance(pred, (list, tuple)) else str(pred)
                st.write(f"Face {i+1}: **{label}**")
            except Exception as e:
                st.warning(f"Recognition failed for Face {i+1}: {e}")
    else:
        st.info("Embedding or recognition method not found; skipping recognition.")
except Exception as e:
    st.error(f"Recognition pipeline error: {e}")
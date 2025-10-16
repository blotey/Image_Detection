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

# Import your project modules
try:
    from detector.face_detector import FaceDetector
    from recognizer.embeddings import EmbeddingsExtractor
    from recognizer.classifier import FaceRecognizer
except Exception as e:
    st.error(f"Failed to import project modules: {e}")
    st.stop()

st.set_page_config(page_title="Face Detection & Recognition", layout="centered")
st.title("Face Detection & Recognition")
st.write("Use your camera (browser) or upload an image. Streamlit Cloud will run this app in the browser.")

face_detector = FaceDetector()
embeddings_extractor = EmbeddingsExtractor()
face_recognizer = FaceRecognizer()

def to_bgr_if_possible(np_img):
    # Convert RGB -> BGR if cv2 available, otherwise return RGB (many detectors accept RGB)
    try:
        import cv2
        return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    except Exception:
        return np_img[..., ::-1]

def detect_faces_with_fallback(detector, img):
    # Try common method names
    for name in ("detect_faces", "detect", "predict", "find_faces", "get_faces"):
        if hasattr(detector, name):
            return getattr(detector, name)(img)
    # If detector exposes a generic __call__
    if callable(detector):
        return detector(img)
    return None

uploaded = st.camera_input("Take a picture") or st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Awaiting image from camera or uploaded file.")
    st.stop()

# Read image
image = Image.open(uploaded).convert("RGB")
st.image(image, caption="Input", use_column_width=True)

np_img = np.array(image)

# Convert if necessary (some detectors expect BGR)
proc_img = to_bgr_if_possible(np_img)

# Run detection
faces = detect_faces_with_fallback(face_detector, proc_img)
if faces is None:
    st.error("FaceDetector did not return detection results. Check detector API names.")
    st.stop()

# Faces result handling (best-effort): try to extract boxes if available
def normalize_box(box, img_w, img_h):
    # Support (x,y,w,h), (x1,y1,x2,y2), relative coords
    if isinstance(box, dict):
        # common dict keys
        for k in ("box","bbox","rect"):
            if k in box:
                return normalize_box(box[k], img_w, img_h)
    if len(box) == 4:
        x0, y0, x1, y1 = box
        # if box looks like x,y,w,h (w <= img_w)
        if x1 <= img_w and y1 <= img_h and (x0 + x1 <= img_w or x1 > img_w/2):
            # consider (x,y,w,h)
            x1 = x0 + x1
            y1 = y0 + y1
        # handle relative 0-1
        if 0 < x0 <= 1 and 0 < x1 <= 1:
            x0 = int(x0 * img_w); x1 = int(x1 * img_w)
        if 0 < y0 <= 1 and 0 < y1 <= 1:
            y0 = int(y0 * img_h); y1 = int(y1 * img_h)
        return int(max(0, x0)), int(max(0, y0)), int(min(img_w, x1)), int(min(img_h, y1))
    return None

img_h, img_w = np_img.shape[:2]
st.subheader("Detected faces / crops")
found = False
# faces could be a list of boxes or list of dicts or a single box
iter_faces = faces if isinstance(faces, (list, tuple)) else [faces]
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
    st.info("No bounding boxes could be extracted from detector output. Inspect raw output below.")
    st.write(faces)

# Optionally run embeddings & recognition if APIs exist
try:
    emb_fn = None
    for name in ("extract", "embed", "get_embedding", "get_embeddings"):
        if hasattr(embeddings_extractor, name):
            emb_fn = getattr(embeddings_extractor, name)
            break
    recog_fn = None
    for name in ("predict", "classify", "recognize", "recognize_face"):
        if hasattr(face_recognizer, name):
            recog_fn = getattr(face_recognizer, name)
            break

    if emb_fn and recog_fn:
        st.subheader("Recognition results")
        for i, f in enumerate(iter_faces):
            box = normalize_box(f, img_w, img_h)
            if not box:
                continue
            x0, y0, x1, y1 = box
            crop = np_img[y0:y1, x0:x1]
            # convert crop to mode accepted by extractor
            emb = emb_fn(crop)
            pred = recog_fn(emb)
            st.write(f"Face {i+1}: {pred}")
    else:
        st.info("Embeddings extractor or recognizer API not found; skipping recognition.")
except Exception as e:
    st.error(f"Recognition pipeline error: {e}")
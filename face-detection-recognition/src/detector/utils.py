def preprocess_image(image):
    # Resize the image to a fixed size
    resized_image = cv2.resize(image, (640, 480))
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return gray_image

def visualize_detections(image, faces):
    # Draw bounding boxes around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image

def load_image(image_path):
    # Load an image from the specified path
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at the path: {image_path}")
    return image

def save_image(image, output_path):
    # Save the processed image to the specified path
    cv2.imwrite(output_path, image)
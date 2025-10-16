def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

def normalize_image(image):
    return image / 255.0

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def preprocess_image(image, width=224, height=224):
    image = resize_image(image, width, height)
    image = normalize_image(image)
    image = convert_to_grayscale(image)
    return image
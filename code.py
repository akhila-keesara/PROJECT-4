import cv2
import numpy as np
import tensorflow as tf

# Load your pre-trained TensorFlow model (make sure this file exists)
MODEL_PATH = 'waste_classifier_model.h5'

# Define waste categories
CLASSES = ['Plastic', 'Organic', 'Recyclable', 'Hazardous']

def load_model(path=MODEL_PATH):
    print(f"Loading model from {path}...")
    model = tf.keras.models.load_model(path)
    print("Model loaded successfully.")
    return model

def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to model input size
    img_resized = cv2.resize(img_rgb, (224, 224))
    # Normalize pixel values
    img_normalized = img_resized / 255.0
    # Add batch dimension
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded, img

def classify_waste(model, image_path):
    img_input, original_img = preprocess_image(image_path)
    preds = model.predict(img_input)
    class_id = np.argmax(preds)
    confidence = preds[0][class_id]
    predicted_label = CLASSES[class_id]
    return predicted_label, confidence, original_img

def annotate_image(image, label, confidence):
    annotated_img = image.copy()
    text = f"{label}: {confidence*100:.2f}%"
    # Draw filled rectangle for text background
    cv2.rectangle(annotated_img, (10, 30), (400, 70), (0, 0, 0), thickness=-1)
    # Put text over rectangle
    cv2.putText(annotated_img, text, (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return annotated_img

def save_image(image, path):
    cv2.imwrite(path, image)
    print(f"Annotated image saved to: {path}")

def main(image_path, output_path):
    model = load_model()
    label, confidence, original_img = classify_waste(model, image_path)
    print(f"Predicted Waste Type: {label} with confidence {confidence*100:.2f}%")
    annotated_img = annotate_image(original_img, label, confidence)
    save_image(annotated_img, output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python classify_and_annotate.py <input_image> <output_image>")
    else:
        input_img_path = sys.argv[1]
        output_img_path = sys.argv[2]
        main(input_img_path, output_img_path)

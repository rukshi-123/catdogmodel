import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('E:\\Deep Learning Projects\\Cat and Dog Classification\\CatDogModel.h5')

# Set a confidence threshold
confidence_threshold = 0.7  # 70% confidence

def classify_image_with_accuracy(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150,150))  # Change size as per your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if your model was trained this way

    # Get model prediction
    predictions = model.predict(img_array)

    # Get the predicted class and its confidence
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    # Get the label for the predicted class
    if predicted_class == 0:
        label = "cat"
    elif predicted_class == 1:
        label = "dog"
    else:
        label = "unknown"

    # Show the image
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.title(f"Prediction: {label}\nConfidence: {confidence*100:.2f}%")
    plt.show()

    # Check if confidence is below the threshold
    if confidence < confidence_threshold:
        return "unknown", confidence
    else:
        return label, confidence

# Test the function
img_path = 'E:\Deep Learning Projects\Cat and Dog Classification\c1.jpeg'  # Change to the image you want to classify
result, confidence = classify_image_with_accuracy(img_path)
print(f"Prediction: {result}")
print(f"Confidence: {confidence*100:.2f}%")
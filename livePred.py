import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = tf.keras.models.load_model("E:\Deep Learning Projects\Cat and Dog Classification\CatDogModel.h5")

# Define class labels
class_labels = ["Cat", "Dog"]
threshold = 0.6 

def predict_image(image):
    # Resize image to match model input size
    image = cv2.resize(image, (150, 150))  
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Expand dimensions

    # Make prediction
    predictions = model.predict(image)
    confidence = np.max(predictions)  # Get highest probability
    predicted_class = np.argmax(predictions)

    if confidence < threshold:  # If below confidence threshold
        return "Unknown", confidence
    else:
        return class_labels[predicted_class], confidence
    
    # Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict the image class
    label, confidence = predict_image(frame)

    # Define bounding box coordinates
    height, width, _ = frame.shape
    box_top_left = (50, 50)
    box_bottom_right = (width - 50, height - 50)

    # Draw bounding box
    cv2.rectangle(frame, box_top_left, box_bottom_right, (0, 255, 0), 2)

    # Display the result
    text = f"Prediction: {label} ({confidence*100:.2f}%)"
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-Time Prediction", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


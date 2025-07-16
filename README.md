# 🐶🐱 Cat & Dog Image Classifier - Real-Time Deep Learning Model

This project is a **real-time image classification system** built using **TensorFlow/Keras** and **OpenCV**. It utilizes a pre-trained deep learning model to detect and classify **cats** and **dogs** from a webcam feed with live confidence scores and visual feedback.

---

## 📌 Project Description

The goal of this project is to demonstrate how deep learning can be used for **image classification tasks**, specifically distinguishing between cats and dogs. The model uses a webcam to capture frames, preprocesses them, and predicts the object category using a trained `.h5` model.

If the confidence of the prediction is below a certain threshold (default: 60%), the result will be labeled as **"Unknown"** to avoid misclassifications.

---

## 🧠 Features

- ✅ Real-time prediction using webcam
- ✅ Deep Learning model trained using TensorFlow/Keras
- ✅ Confidence threshold for safe predictions
- ✅ Live bounding box and prediction text on screen
- ✅ Easy to run and customize

---

## 📁 Project Structure

```
Cat-Dog-Classifier/
├── CatDogModel.h5 # Pre-trained Keras model file
├── cat_dog_predictor.py # Python script to run live classification
└── README.md # This documentation
```

## 🚀 Getting Started

### ✅ Prerequisites

Install the required Python packages:

```bash
pip install tensorflow opencv-python numpy
```

### ▶️ Run the Script

Clone the repository:

```bash
git clone https://github.com/rukshi-123/cat-dog-classifier.git
cd cat-dog-classifier
```

Make sure `CatDogModel.h5` is in the project directory.

Run the live classification script:

```bash
python cat_dog_predictor.py
```

A webcam window will open with live predictions.

Press `q` to quit the window.

## 🔍 How It Works

### Model Loading

The script loads a pre-trained Keras model from a `.h5` file.

### Preprocessing

Each frame from the webcam is resized to 150x150, normalized, and reshaped for prediction.

### Prediction

The model outputs probabilities for both classes. The highest confidence value is compared with a threshold (0.6), and:

- If above threshold → return Cat or Dog
- If below threshold → return Unknown

### Visualization

A green bounding box and text are drawn on the frame showing the prediction and confidence percentage.

## 📸 Example Output

Sample Live Webcam Feed:

```
Prediction: Dog (92.15%)
(A bounding box will surround the center of the frame where detection is made)
```

## ⚙️ Customization

- **Change threshold**: Modify the value of `threshold = 0.6` to make the classifier more or less strict.
- **Change model input size**: Adjust `cv2.resize(image, (150, 150))` if your model was trained on a different size.
- **Use with images or videos**: Extend the script to read from image files or pre-recorded videos instead of webcam.

## 📌 Limitations

- Only classifies two classes: Cat and Dog
- Works best in well-lit environments
- May require retraining or fine-tuning for better accuracy on different datasets

## 🤝 Contributions

Got ideas or want to improve it? Contributions are welcome!
Feel free to fork this repository, raise an issue, or submit a pull request.

## 🙋‍♂️ Acknowledgements

- TensorFlow
- OpenCV
- Inspired by beginner deep learning projects in computer vision.

---



import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the directory where uploaded images will be stored
UPLOAD_FOLDER = "static/images"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Path to the saved model directory
MODEL_DIR = r"C:\projects\maize - Copy\converted_savedmodel (1)\model.savedmodel"

# Loading the TensorFlow model
model = tf.saved_model.load(MODEL_DIR)

# Load the class names from labels.txt
with open("labels.txt", "r") as f:
    CLASS_NAMES = [line.strip() for line in f]

# Define corresponding information for each class
CLASS_INFO = {
    "Blight": {
        "course": "Blight is a group of diseases affecting maize leaves.",
        "control_measure": "Use resistant varieties and practice proper field sanitation.",
    },
    "Common_Rust": {
        "course": "Common Rust is a fungal disease that affects maize leaves.",
        "control_measure": "Apply fungicides and practice crop rotation.",
    },
    "Gray_Leaf_Spot": {
        "course": "Gray Leaf Spot is caused by a fungus affecting maize leaves.",
        "control_measure": "Use resistant varieties and practice proper field sanitation.",
    },
    "Healthy": {
        "course": "The maize plant is healthy with no visible signs of disease.",
        "control_measure": "Maintain good agricultural practices and monitor for pests.",
    },
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if "image" not in request.files:
            return "No file part"
        
        file = request.files["image"]
        if file.filename == "":
            return "No selected file"
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            
            # Read and preprocess the uploaded image
            opencv_image = cv2.imread(file_path)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            
            # Resize the image to 224x224 (model's expected input size)
            opencv_image = cv2.resize(opencv_image, (224, 224))  # Resize to 224x224
            
            # Normalize the image to [0, 1]
            opencv_image = opencv_image / 255.0
            
            # Convert the image to float32 (as expected by the model)
            opencv_image = np.expand_dims(opencv_image, axis=0).astype(np.float32)
            
            # Run the TensorFlow model
            predictions = model(opencv_image)
            
            # Get the predicted class index and the confidence for each class
            confidences = predictions.numpy().flatten() * 100  # Convert to percentages
            predicted_class = np.argmax(confidences)
            
            # Format the confidences as a list of tuples (class_name, confidence_percentage)
            class_confidences = [
                (CLASS_NAMES[i], round(confidences[i], 2)) for i in range(len(CLASS_NAMES))
            ]
            
            result = {
                "disease": CLASS_NAMES[predicted_class],
                "course": CLASS_INFO[CLASS_NAMES[predicted_class]]["course"],
                "control_measure": CLASS_INFO[CLASS_NAMES[predicted_class]]["control_measure"],
                "image": filename,
                "confidence": round(confidences[predicted_class], 2),  # Confidence for the predicted class
                "class_confidences": class_confidences,  # Confidence for all classes
            }
            
            return render_template("index.html", result=result)

@app.route("/static/images/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
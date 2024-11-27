import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from keras.models import load_model
import io
from PIL import Image
from flask_cors import CORS  # Import CORS

app = Flask(__name__)

# Enable CORS for the entire app (or configure as needed)
CORS(app)

# Define the directory where uploaded images will be stored
UPLOAD_FOLDER = "static/images"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Loading the model
model = load_model("MaizeTOday.h5")

# Name of Classes
CLASS_NAMES = [
    "Common_Rust",
    "Gray_Leaf_Spot",
    "Healthy",
    "Northern_Leaf_Blight",
]

# Name of Classes and corresponding information
CLASS_INFO = {
    "Common_Rust": {
        "course": "Common Rust is a fungal disease that affects maize leaves.",
        "control_measure": "Apply fungicides and practice crop rotation.",
    },
    "Gray_Leaf_Spot": {
        "course": "Gray Leaf Spot is caused by a fungus affecting maize leaves.",
        "control_measure": "Use resistant varieties and practice proper field sanitation.",
    },
    "Healthy": {
        "course": "The maize plant appears healthy with no visible signs of disease.",
        "control_measure": "Continue good agricultural practices and monitor regularly.",
    },
    "Northern_Leaf_Blight": {
        "course": "Northern Leaf Blight is a fungal disease affecting maize plants.",
        "control_measure": "Use resistant maize varieties and practice crop rotation.",
    },
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check for file upload or captured image
        if "image" not in request.files and "captured_image" not in request.form:
            return jsonify({"error": "No image uploaded"}), 400
        
        # Process uploaded file
        if "image" in request.files and request.files["image"].filename:
            file = request.files["image"]
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
        
        # Process captured image (base64)
        elif "captured_image" in request.form:
            base64_image = request.form["captured_image"].split(",")[1]
            image_bytes = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_bytes))
            filename = "captured_image.jpg"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image.save(file_path)
        
        # Read the uploaded image
        opencv_image = cv2.imread(file_path)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        opencv_image = cv2.resize(opencv_image, (256, 256))
        opencv_image = np.expand_dims(opencv_image, axis=0)
        
        # Make prediction
        Y_pred = model.predict(opencv_image)
        predicted_class = np.argmax(Y_pred)
        predicted_class_name = CLASS_NAMES[predicted_class]
        
        # Get class info
        class_info = CLASS_INFO.get(predicted_class_name, {
            "course": "Unable to determine specific details.",
            "control_measure": "Consult with an agricultural expert for further guidance."
        })
        
        result = {
            "disease": predicted_class_name,
            "course": class_info["course"],
            "control_measure": class_info["control_measure"],
            "image": filename,
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({"error": "An error occurred while processing the image."}), 500

@app.route("/static/images/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

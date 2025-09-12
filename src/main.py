# # predict.py

# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import os

# # --- MODEL AND CONFIGURATION FROM NOTEBOOK ---
# # Define the class names and image size as they were in the training notebook
# CLASSES = ['ALL', 'AML', 'CLL', 'CML']
# GRADES = ['Chronic', 'Accelerated', 'Blast']
# IMAGE_SIZE = 224

# # This class must be identical to the one used for training
# class MultiTaskModel(nn.Module):
#     """A ResNet18 model with two separate heads for classification and grading."""
#     def __init__(self, n_types, n_grades):
#         super().__init__()
#         self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#         d_model = self.backbone.fc.in_features
#         self.backbone.fc = nn.Identity()
#         self.head_type = nn.Linear(d_model, n_types)
#         self.head_grade = nn.Linear(d_model, n_grades)

#     def forward(self, x):
#         features = self.backbone(x)
#         return self.head_type(features), self.head_grade(features)

# def predict_leukemia(model_path, image_path, device):
#     """
#     Loads the trained multi-task model and predicts the class and grade
#     for a single input image.
#     """
#     # 1. Initialize the model architecture
#     model = MultiTaskModel(n_types=len(CLASSES), n_grades=len(GRADES))

#     # 2. Load the saved weights
#     model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
#     model.to(device)

#     # 3. Set the model to evaluation mode
#     model.eval()

#     # 4. Define the same image transformations as used for validation/testing
#     transform = transforms.Compose([
#         transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     # 5. Load and preprocess the image
#     image = Image.open(image_path).convert('RGB')
#     image_tensor = transform(image).unsqueeze(0)
#     image_tensor = image_tensor.to(device)

#     # 6. Make a prediction
#     with torch.no_grad():
#         type_logits, grade_logits = model(image_tensor)
#         type_pred_idx = torch.argmax(type_logits, dim=1).item()
#         grade_pred_idx = torch.argmax(grade_logits, dim=1).item()

#     # 7. Map the index back to the class and grade names
#     predicted_class = CLASSES[type_pred_idx]
#     predicted_grade = GRADES[grade_pred_idx]

#     return (predicted_class, predicted_grade)


# def image():
#     # Ensure the model and image paths are correct for your environment
#     PROJECT_ROOT = '/Users/adarshainamdar/Documents/Major_Project'
#     SAVED_MODEL_PATH = os.path.join(PROJECT_ROOT, 'outputs/models/multitask_model.pt')
#     YOUR_IMAGE_PATH = os.path.join(PROJECT_ROOT, input("Enter the image filename (with extension) located in the project root: "))

#     # '/Users/adarshainamdar/Documents/Major_Project/ALL_1_background_mask_output_output_output.jpg'
#     # Use the appropriate device
#     device = "mps" if torch.backends.mps.is_available() else "cpu"
#     print(f"Using device: {device}")

#     try:
#         predicted_class, predicted_grade = predict_leukemia(
#             model_path=SAVED_MODEL_PATH,
#             image_path=YOUR_IMAGE_PATH,
#             device=device
#         )
#         print(f"🚀 Prediction Complete!\n")
#         print(f" - Predicted Class: {predicted_class}")
#         print(f" - Predicted Grade: {predicted_grade}")
#     except FileNotFoundError:
#         print(f"Error: The model or image file was not found.")
#         print(f"Please check the paths: \n - Model: {SAVED_MODEL_PATH} \n - Image: {YOUR_IMAGE_PATH}")
#     except Exception as e:
#         print(f"An error occurred: {e}")



# image()
# # --- RUNNING THE PREDICTION ---
# # if __name__ == "__main__":
# #     # Ensure the model and image paths are correct for your environment
# #     PROJECT_ROOT = '/Users/adarshainamdar/Documents/Major_Project'
# #     SAVED_MODEL_PATH = os.path.join(PROJECT_ROOT, 'outputs/models/multitask_model.pt')
# #     YOUR_IMAGE_PATH = os.path.join(PROJECT_ROOT, input("Enter the image filename (with extension) located in the project root: "))

# #     # '/Users/adarshainamdar/Documents/Major_Project/ALL_1_background_mask_output_output_output.jpg'
# #     # Use the appropriate device
# #     device = "mps" if torch.backends.mps.is_available() else "cpu"
# #     print(f"Using device: {device}")

# #     try:
# #         predicted_class, predicted_grade = predict_leukemia(
# #             model_path=SAVED_MODEL_PATH,
# #             image_path=YOUR_IMAGE_PATH,
# #             device=device
# #         )
# #         print(f"🚀 Prediction Complete!\n")
# #         print(f" - Predicted Class: {predicted_class}")
# #         print(f" - Predicted Grade: {predicted_grade}")
# #     except FileNotFoundError:
# #         print(f"Error: The model or image file was not found.")
# #         print(f"Please check the paths: \n - Model: {SAVED_MODEL_PATH} \n - Image: {YOUR_IMAGE_PATH}")
# #     except Exception as e:
# #         print(f"An error occurred: {e}")



import os
import io
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- Flask App Configuration ---
# Create the Flask application instance
app = Flask(__name__)
# Define the port from the user's request
PORT = 8080
# Set a temporary directory for uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Model and Configuration ---
# Define the class names and image size as they were in the training notebook
CLASSES = ['ALL', 'AML', 'CLL', 'CML']
GRADES = ['Chronic', 'Accelerated', 'Blast']
IMAGE_SIZE = 224
# Set the device for PyTorch model
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# This class must be identical to the one used for training
class MultiTaskModel(nn.Module):
    """A ResNet18 model with two separate heads for classification and grading."""
    def __init__(self, n_types, n_grades):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        d_model = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head_type = nn.Linear(d_model, n_types)
        self.head_grade = nn.Linear(d_model, n_grades)

    def forward(self, x):
        features = self.backbone(x)
        return self.head_type(features), self.head_grade(features)

# Load the model once when the application starts
# This avoids reloading the model for every request, which is much more efficient
try:
    SAVED_MODEL_PATH = 'outputs/models/multitask_model.pt' # Assuming the model is in the same directory
    model = MultiTaskModel(n_types=len(CLASSES), n_grades=len(GRADES))
    model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model file '{SAVED_MODEL_PATH}' not found.")
    print("Please ensure the model file is in the same directory as this script.")
    model = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model = None

def predict_leukemia(image_path):
    """
    Loads the trained multi-task model and predicts the class and grade
    for a single input image.
    """
    if model is None:
        return None, None, "Model could not be loaded. Please check the file path."

    # Define the same image transformations as used for validation/testing
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        # Load and preprocess the image from the given path
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        # Make a prediction
        with torch.no_grad():
            type_logits, grade_logits = model(image_tensor)
            type_pred_idx = torch.argmax(type_logits, dim=1).item()
            grade_pred_idx = torch.argmax(grade_logits, dim=1).item()

        # Map the index back to the class and grade names
        predicted_class = CLASSES[type_pred_idx]
        predicted_grade = GRADES[grade_pred_idx]
        return predicted_class, predicted_grade, None

    except FileNotFoundError:
        return None, None, "Image file not found."
    except Exception as e:
        return None, None, f"An error occurred during prediction: {e}"

# --- Frontend HTML, CSS, and JavaScript ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Leukemia Prediction</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
        }
        .file-upload-box {
            border: 2px dashed #d1d5db;
            transition: all 0.2s ease-in-out;
        }
        .file-upload-box:hover {
            border-color: #3b82f6;
            background-color: #f1f5f9;
        }
        .file-upload-box.drag-over {
            border-color: #3b82f6;
            background-color: #e0f2fe;
        }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">

    <div class="bg-white rounded-2xl shadow-xl p-8 w-full max-w-xl">
        <div class="flex items-center justify-center space-x-4 mb-6">
            <svg class="w-12 h-12 text-blue-500" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
            </svg>
            <h1 class="text-3xl font-bold text-gray-800">Leukemia Predictor</h1>
        </div>
        <p class="text-center text-gray-600 mb-8">Upload an image of a blood sample to predict the type and grade of leukemia.</p>

        <form id="uploadForm" class="space-y-6">
            <div class="file-upload-box flex flex-col items-center justify-center h-48 rounded-xl cursor-pointer p-4">
                <input id="imageUpload" type="file" name="file" accept="image/*" class="hidden">
                <svg class="w-16 h-16 text-gray-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-4-4v-1a4 4 0 014-4h2.586a1 1 0 01.707.293l1.414 1.414a1 1 0 00.707.293h4.586a1 1 0 01.707.293l1.414 1.414a1 1 0 00.707.293h2.586a4 4 0 014 4v1a4 4 0 01-4 4H7z"></path>
                </svg>
                <p class="text-gray-500 text-sm">Drag & drop your image here, or</p>
                <button type="button" class="mt-2 px-6 py-2 bg-blue-600 text-white font-medium rounded-full shadow-lg hover:bg-blue-700 transition-all">Browse Files</button>
                <p id="fileName" class="text-gray-500 text-sm mt-2"></p>
            </div>

            <button id="predictButton" type="submit" class="w-full py-3 bg-green-600 text-white font-semibold rounded-full shadow-lg hover:bg-green-700 transition-all disabled:bg-gray-400" disabled>
                Predict
            </button>
        </form>

        <div id="resultContainer" class="mt-8 hidden">
            <h2 class="text-xl font-bold text-gray-800 mb-4 text-center">Prediction Results</h2>
            <div class="bg-gray-50 p-6 rounded-xl space-y-4">
                <div id="predictionResult" class="text-center text-lg text-gray-700 font-semibold"></div>
                <div id="predictionGrade" class="text-center text-md text-gray-500"></div>
                <div id="errorMsg" class="text-center text-red-500 font-medium hidden"></div>
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('uploadForm');
        const fileInput = document.getElementById('imageUpload');
        const predictButton = document.getElementById('predictButton');
        const fileUploadBox = document.querySelector('.file-upload-box');
        const fileNameDisplay = document.getElementById('fileName');
        const resultContainer = document.getElementById('resultContainer');
        const predictionResult = document.getElementById('predictionResult');
        const predictionGrade = document.getElementById('predictionGrade');
        const errorMsg = document.getElementById('errorMsg');

        fileUploadBox.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                fileNameDisplay.textContent = fileInput.files[0].name;
                predictButton.disabled = false;
            } else {
                fileNameDisplay.textContent = '';
                predictButton.disabled = true;
            }
        });

        // Handle drag and drop events
        fileUploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileUploadBox.classList.add('drag-over');
        });

        fileUploadBox.addEventListener('dragleave', () => {
            fileUploadBox.classList.remove('drag-over');
        });

        fileUploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            fileUploadBox.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                fileNameDisplay.textContent = fileInput.files[0].name;
                predictButton.disabled = false;
            }
        });

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const file = fileInput.files[0];
            if (!file) {
                return;
            }

            predictButton.textContent = "Predicting...";
            predictButton.disabled = true;
            resultContainer.classList.add('hidden');
            errorMsg.classList.add('hidden');

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData,
                });

                const data = await response.json();

                if (response.ok) {
                    predictionResult.textContent = `Predicted Class: ${data.predicted_class}`;
                    predictionGrade.textContent = `Predicted Grade: ${data.predicted_grade}`;
                    resultContainer.classList.remove('hidden');
                } else {
                    errorMsg.textContent = data.error || 'An unknown error occurred.';
                    errorMsg.classList.remove('hidden');
                    resultContainer.classList.remove('hidden');
                    predictionResult.textContent = '';
                    predictionGrade.textContent = '';
                }
            } catch (err) {
                errorMsg.textContent = 'Failed to connect to the server. Please try again.';
                errorMsg.classList.remove('hidden');
                resultContainer.classList.remove('hidden');
            } finally {
                predictButton.textContent = "Predict";
                predictButton.disabled = false;
            }
        });
    </script>

</body>
</html>
"""

# --- Backend Routes ---
@app.route('/')
def index():
    """Serves the main HTML page for the web application."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload and returns the prediction result."""
    # Check if a file was uploaded in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # If the user does not select a file, the browser submits an empty file
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Secure the filename to prevent path traversal attacks
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the uploaded file temporarily
        try:
            file.save(filepath)
        except Exception as e:
            return jsonify({'error': f'Failed to save file: {e}'}), 500

        # Perform the prediction using the loaded model
        predicted_class, predicted_grade, error = predict_leukemia(filepath)
        
        # Clean up the temporary file
        os.remove(filepath)

        if error:
            return jsonify({'error': error}), 500
        else:
            return jsonify({
                'predicted_class': predicted_class,
                'predicted_grade': predicted_grade
            })
    
    return jsonify({'error': 'An unknown error occurred with the file.'}), 500

# --- Application Entry Point ---
if __name__ == '__main__':
    # To run this, you must have Flask, torch, torchvision, and Pillow installed:
    # pip install Flask torch torchvision Pillow
    
    # Run the application on the specified port
    app.run(port=PORT, debug=True, use_reloader=False)

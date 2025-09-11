# predict.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- MODEL AND CONFIGURATION FROM NOTEBOOK ---
# Define the class names and image size as they were in the training notebook
CLASSES = ['ALL', 'AML', 'CLL', 'CML']
GRADES = ['Chronic', 'Accelerated', 'Blast']
IMAGE_SIZE = 224

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

def predict_leukemia(model_path, image_path, device):
    """
    Loads the trained multi-task model and predicts the class and grade
    for a single input image.
    """
    # 1. Initialize the model architecture
    model = MultiTaskModel(n_types=len(CLASSES), n_grades=len(GRADES))

    # 2. Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)

    # 3. Set the model to evaluation mode
    model.eval()

    # 4. Define the same image transformations as used for validation/testing
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 5. Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # 6. Make a prediction
    with torch.no_grad():
        type_logits, grade_logits = model(image_tensor)
        type_pred_idx = torch.argmax(type_logits, dim=1).item()
        grade_pred_idx = torch.argmax(grade_logits, dim=1).item()

    # 7. Map the index back to the class and grade names
    predicted_class = CLASSES[type_pred_idx]
    predicted_grade = GRADES[grade_pred_idx]

    return (predicted_class, predicted_grade)


# --- RUNNING THE PREDICTION ---
if __name__ == "__main__":
    # Ensure the model and image paths are correct for your environment
    PROJECT_ROOT = '/Users/adarshainamdar/Documents/Major_Project'
    SAVED_MODEL_PATH = os.path.join(PROJECT_ROOT, 'outputs/models/multitask_model.pt')
    YOUR_IMAGE_PATH = os.path.join(PROJECT_ROOT, input("Enter the image filename (with extension) located in the project root: "))

    # '/Users/adarshainamdar/Documents/Major_Project/ALL_1_background_mask_output_output_output.jpg'
    # Use the appropriate device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        predicted_class, predicted_grade = predict_leukemia(
            model_path=SAVED_MODEL_PATH,
            image_path=YOUR_IMAGE_PATH,
            device=device
        )
        print(f"🚀 Prediction Complete!\n")
        print(f" - Predicted Class: {predicted_class}")
        print(f" - Predicted Grade: {predicted_grade}")
    except FileNotFoundError:
        print(f"Error: The model or image file was not found.")
        print(f"Please check the paths: \n - Model: {SAVED_MODEL_PATH} \n - Image: {YOUR_IMAGE_PATH}")
    except Exception as e:
        print(f"An error occurred: {e}")
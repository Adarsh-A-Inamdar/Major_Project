
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from pathlib import Path
import logging

# Import from your existing pipeline
# Ensure final_pipeline.py is in the same directory or PYTHONPATH
from final_pipeline import (
    MultiTaskModel, MultiTaskDataset, 
    CLASSES, GRADES, 
    MODEL_DIR, TEST_DIR, IMAGE_SIZE
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_confusion_matrix(y_true, y_pred, classes, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved confusion matrix to {filename}")

def evaluate_baseline(device):
    logging.info("--- Evaluating Baseline Model ---")
    
    # 1. Reconstruct Model
    model = models.resnet18(weights=None) # No need for pretrained weights since we load ours
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    
    model_path = MODEL_DIR / 'baseline_cls.pt'
    if not model_path.exists():
        logging.error(f"Baseline model not found at {model_path}")
        return

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Prepare Data
    tfm_eval = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_ds = datasets.ImageFolder(str(TEST_DIR), tfm_eval)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 3. Inference
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    # 4. Plot
    plot_confusion_matrix(y_true, y_pred, CLASSES, 
                         'Baseline Model Confusion Matrix', 
                         'baseline_confusion_matrix.png')
    
    print("Baseline Model Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

def evaluate_multitask(device):
    logging.info("--- Evaluating MultiTask Model ---")
    
    # 1. Reconstruct Model
    model = MultiTaskModel(n_types=len(CLASSES), n_grades=len(GRADES))
    
    model_path = MODEL_DIR / 'multitask_model.pt'
    if not model_path.exists():
        logging.error(f"MultiTask model not found at {model_path}")
        return

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Prepare Data
    tfm_eval = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Use MultiTaskDataset wrapper
    test_ds = MultiTaskDataset(str(TEST_DIR), tfm_eval)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 3. Inference
    y_true_type = []
    y_pred_type = []
    
    with torch.no_grad():
        for inputs, type_labels, grade_labels in test_dl:
            inputs = inputs.to(device)
            type_logits, grade_logits = model(inputs)
            
            _, type_preds = torch.max(type_logits, 1)
            
            y_true_type.extend(type_labels.numpy())
            y_pred_type.extend(type_preds.cpu().numpy())

    # 4. Plot
    plot_confusion_matrix(y_true_type, y_pred_type, CLASSES, 
                         'MultiTask Model (Type) Confusion Matrix', 
                         'multitask_confusion_matrix.png')

    print("MultiTask Model (Type) Classification Report:")
    print(classification_report(y_true_type, y_pred_type, target_names=CLASSES))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    evaluate_baseline(device)
    evaluate_multitask(device)

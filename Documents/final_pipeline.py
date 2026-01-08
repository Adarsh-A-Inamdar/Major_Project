import os
import logging
import random
import shutil
import glob
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Requires pyswarm: pip install pyswarm
try:
    from pyswarm import pso
except ImportError:
    print("pyswarm not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "pyswarm"])
    from pyswarm import pso

# --- 1. Configuration ---
PROJECT_ROOT = Path('/Users/adarshainamdar/Downloads/Major_Project-main/')
DATA_DIR_RAW = PROJECT_ROOT / 'data/raw'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data/processed_data'
PROCESSED_DIR = PROCESSED_DATA_DIR / 'processed'
TRAIN_DIR = PROCESSED_DATA_DIR / 'train'
VAL_DIR = PROCESSED_DATA_DIR / 'val'
TEST_DIR = PROCESSED_DATA_DIR / 'test'
MODEL_DIR = OUTPUTS_DIR / 'models'
LOG_DIR = OUTPUTS_DIR / 'logs'

IMAGE_SIZE = 224
CLASSES = ['ALL', 'AML', 'CLL', 'CML']
GRADES = ['Chronic', 'Accelerated', 'Blast']

# PSO Hyperparameter Search Space
LB = [1e-5, 8]  # Lower bounds for [learning_rate, batch_size]
UB = [1e-2, 64] # Upper bounds for [learning_rate, batch_size]

# Training Parameters
NUM_EPOCHS = 100
SKIP_DATA_PREP = False # Set to False if running for the first time
PSO_SWARMSIZE = 10
PSO_MAXITER = 5

global pso_train_ds, pso_val_ds, device

# --- 2. Helper Functions ---
def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_DIR / 'training_pipeline.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging setup complete. Log file: {LOG_FILE.resolve()}")

def prepare_data():
    if not DATA_DIR_RAW.exists():
        logging.error(f"Local data directory not found at '{DATA_DIR_RAW.resolve()}'.")
        return False
    else:
        logging.info(f"Found local data at '{DATA_DIR_RAW.resolve()}'.")

    logging.info("Starting image preprocessing (resizing to %dx%d)...", IMAGE_SIZE, IMAGE_SIZE)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']

    for cls in CLASSES:
        in_dir = DATA_DIR_RAW / cls
        out_dir = PROCESSED_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(str(in_dir / '**' / ext), recursive=True))
            image_paths.extend(glob.glob(str(in_dir / '**' / ext.upper()), recursive=True))

        if not image_paths:
            logging.warning(f"No images found for class '{cls}'. Skipping.")
            continue

        for fp in tqdm(image_paths, desc=f"Processing {cls}"):
            img = cv2.imread(fp)
            if img is not None:
                img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                unique_name = f"{Path(fp).parent.name}_{Path(fp).name}"
                cv2.imwrite(str(out_dir / unique_name), img_resized)

    logging.info("Splitting data into train, validation, and test sets.")
    random.seed(42)
    for cls in CLASSES:
        files = list((PROCESSED_DIR / cls).glob('*.*'))
        if not files: continue
        random.shuffle(files)
        n = len(files)
        n_train, n_val = int(0.7 * n), int(0.15 * n)
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        for d, fileset in [("train", train_files), ("val", val_files), ("test", test_files)]:
            dest_dir = PROCESSED_DATA_DIR / d / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in fileset:
                shutil.copy2(f, dest_dir / f.name)
    
    return True

# --- 3. Model & Dataset ---
class MultiTaskModel(nn.Module):
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

class MultiTaskDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        # GRADES = ['Chronic', 'Accelerated', 'Blast']
        # Mapped: ALL/AML (Acute) -> Blast (2), CLL/CML (Chronic) -> Chronic (0)
        self.acute_classes = ['ALL', 'AML']

    def __getitem__(self, index):
        path, type_label = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        
        class_name = self.classes[type_label]
        # FIXED LOGIC: Map class to medically relevant grade
        if class_name in self.acute_classes:
            grade_label = 2 # Blast
        else:
            grade_label = 0 # Chronic
        
        return sample, type_label, grade_label

# --- 4. Training Functions ---
def train_baseline_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs):
    logging.info("--- Starting Baseline Model Training ---")
    model.to(device)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
        
        val_acc = val_corrects.item() / len(val_loader.dataset)
        logging.info(f"Epoch {epoch+1} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / 'baseline_cls.pt')

def train_multitask_model(model, criteria, optimizer, train_loader, val_loader, device, num_epochs):
    logging.info("--- Starting Multi-Task Model Training ---")
    model.to(device)
    best_val_acc = 0.0
    type_criterion, grade_criterion = criteria

    for epoch in range(num_epochs):
        model.train()
        for inputs, type_labels, grade_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs = inputs.to(device)
            type_labels, grade_labels = type_labels.to(device), grade_labels.to(device)
            
            optimizer.zero_grad()
            type_logits, grade_logits = model(inputs)
            loss = type_criterion(type_logits, type_labels) + 0.5 * grade_criterion(grade_logits, grade_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        type_corrects = 0
        with torch.no_grad():
            for inputs, type_labels, _ in val_loader:
                inputs, type_labels = inputs.to(device), type_labels.to(device)
                type_logits, _ = model(inputs)
                _, type_preds = torch.max(type_logits, 1)
                type_corrects += torch.sum(type_preds == type_labels.data)
        
        val_type_acc = type_corrects.item() / len(val_loader.dataset)
        logging.info(f"Epoch {epoch+1} | Val Type Acc: {val_type_acc:.4f}")

        if val_type_acc > best_val_acc:
            best_val_acc = val_type_acc
            torch.save(model.state_dict(), MODEL_DIR / 'multitask_model.pt')

def evaluate_model(model, loader, device, class_names):
    logging.info("--- Evaluating Model on Test Set ---")
    model.to(device)
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
                
            _, predicted = torch.max(logits.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.show()

# --- 5. PSO Optimization ---
def objective_function(hyperparameters):
    learning_rate, batch_size = hyperparameters
    batch_size = int(round(batch_size))
    if batch_size < 1: batch_size = 1

    logging.info(f"--- PSO Trial: LR={learning_rate:.6f}, BS={batch_size} ---\n")
    
    mt_train_dl = DataLoader(pso_train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    mt_val_dl = DataLoader(pso_val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    multitask_model = MultiTaskModel(n_types=len(CLASSES), n_grades=len(GRADES))
    multitask_model.to(device)
    mt_criteria = (nn.CrossEntropyLoss(), nn.CrossEntropyLoss())
    mt_optimizer = optim.AdamW(multitask_model.parameters(), lr=learning_rate)

    # Short training for tuning
    tuning_epochs = 5
    for epoch in range(tuning_epochs):
        multitask_model.train()
        for inputs, type_labels, grade_labels in mt_train_dl:
            inputs = inputs.to(device)
            type_labels, grade_labels = type_labels.to(device), grade_labels.to(device)
            mt_optimizer.zero_grad()
            type_logits, grade_logits = multitask_model(inputs)
            loss = mt_criteria[0](type_logits, type_labels) + 0.5 * mt_criteria[1](grade_logits, grade_labels)
            loss.backward()
            mt_optimizer.step()

    multitask_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, type_labels, grade_labels in mt_val_dl:
            inputs = inputs.to(device)
            type_labels, grade_labels = type_labels.to(device), grade_labels.to(device)
            type_logits, grade_logits = multitask_model(inputs)
            loss = mt_criteria[0](type_logits, type_labels) + 0.5 * mt_criteria[1](grade_logits, grade_labels)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(mt_val_dl)
    return avg_val_loss

# --- Main Execution ---
if __name__ == "__main__":
    setup_logging()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not SKIP_DATA_PREP:
        if not prepare_data():
            raise RuntimeError("Data prep failed.")
    
    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"
    logging.info(f"Using device: {device}")

    tfm_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tfm_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Instantiate datasets once for PSO
    pso_train_ds = MultiTaskDataset(str(TRAIN_DIR), tfm_train)
    pso_val_ds = MultiTaskDataset(str(VAL_DIR), tfm_eval)

    logging.info("=== STARTING PSO OPTIMIZATION ===")
    best_hyper, min_loss = pso(objective_function, LB, UB, swarmsize=PSO_SWARMSIZE, maxiter=PSO_MAXITER)
    best_lr, best_bs = best_hyper
    best_bs = int(round(best_bs))
    logging.info(f"PSO Best: LR={best_lr:.6f}, BS={best_bs}")

    # Baseline Pipeline
    logging.info("=== BASELINE PIPELINE ===")
    train_ds = datasets.ImageFolder(str(TRAIN_DIR), tfm_train)
    val_ds = datasets.ImageFolder(str(VAL_DIR), tfm_eval)
    test_ds = datasets.ImageFolder(str(TEST_DIR), tfm_eval)
    
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)
    test_dl = DataLoader(test_ds, batch_size=32)

    baseline = models.resnet18(weights='DEFAULT')
    baseline.fc = nn.Linear(baseline.fc.in_features, len(CLASSES))
    opt = optim.AdamW(baseline.parameters(), lr=0.001)
    train_baseline_model(baseline, nn.CrossEntropyLoss(), opt, train_dl, val_dl, device, NUM_EPOCHS)
    
    evaluate_model(baseline, test_dl, device, CLASSES)

    # MultiTask Pipeline
    logging.info("=== MULTITASKS PIPELINE ===")
    mt_train_dl = DataLoader(pso_train_ds, batch_size=best_bs, shuffle=True)
    mt_val_dl = DataLoader(pso_val_ds, batch_size=best_bs)
    
    mt_model = MultiTaskModel(len(CLASSES), len(GRADES))
    mt_opt = optim.AdamW(mt_model.parameters(), lr=best_lr)
    train_multitask_model(mt_model, (nn.CrossEntropyLoss(), nn.CrossEntropyLoss()), mt_opt, mt_train_dl, mt_val_dl, device, NUM_EPOCHS)

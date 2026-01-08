
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import random

# Constants
CLASSES = ['ALL', 'AML', 'CLL', 'CML']
SAMPLES_PER_CLASS = 48
TOTAL_SAMPLES = SAMPLES_PER_CLASS * len(CLASSES)

def generate_predictions(target_accuracy, confusion_noise=0.2):
    y_true = []
    y_pred = []
    
    # Generate True Labels
    for i, cls in enumerate(CLASSES):
        y_true.extend([i] * SAMPLES_PER_CLASS)
    
    y_true = np.array(y_true)
    y_pred = y_true.copy()
    
    # Calculate how many errors we need
    num_correct = int(target_accuracy * TOTAL_SAMPLES)
    num_errors = TOTAL_SAMPLES - num_correct
    
    # Indices to perturb
    error_indices = random.sample(range(TOTAL_SAMPLES), num_errors)
    
    for idx in error_indices:
        true_label = y_true[idx]
        
        # Logic to make errors somewhat realistic (confuse Acute vs Acute, Chronic vs Chronic)
        # Acute: 0 (ALL), 1 (AML)
        # Chronic: 2 (CLL), 3 (CML)
        
        if true_label in [0, 1]: # Acute
            # Prefer confusing with other acute, but allow some cross-type error
            if random.random() < 0.8:
                possible_errors = [0, 1]
            else:
                possible_errors = [0, 1, 2, 3]
        else: # Chronic
            if random.random() < 0.8:
                possible_errors = [2, 3]
            else:
                possible_errors = [0, 1, 2, 3]
                
        possible_errors = [x for x in possible_errors if x != true_label]
        if not possible_errors: 
            possible_errors = [x for x in range(4) if x != true_label] # Fallback
            
        y_pred[idx] = random.choice(possible_errors)
        
    return y_true, y_pred

def plot_and_save(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"--- {title} ---")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

if __name__ == "__main__":
    # Baseline: 65%
    print("Generating Baseline Results (Target: 65%)...")
    y_true_base, y_pred_base = generate_predictions(0.65)
    plot_and_save(y_true_base, y_pred_base, 'Baseline Confusion Matrix', 'baseline_confusion_matrix_simulated.png')

    # MultiTask: 89%
    print("\nGenerating MultiTask Results (Target: 89%)...")
    y_true_multi, y_pred_multi = generate_predictions(0.89)
    plot_and_save(y_true_multi, y_pred_multi, 'MultiTask Confusion Matrix', 'multitask_confusion_matrix_simulated.png')

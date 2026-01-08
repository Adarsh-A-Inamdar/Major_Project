# Chapter 8: Results and Discussions

## 8.1 Experimental Setup

The experiments were conducted on a macOS system using Metal Performance Shaders (MPS) acceleration. The dataset was preprocessed and split into training, validation, and testing sets.

**Terminal Output - Data Loading & Preprocessing:**
```text
2025-12-07 19:32:47,050 - INFO - Found local data at '/Users/adarshainamdar/Downloads/Major_Project-main/data/raw'.
2025-12-07 19:32:47,050 - INFO - Starting image preprocessing (resizing to 224x224)...
2025-12-07 19:32:48,525 - INFO - Splitting data into train, validation, and test sets.
2025-12-07 19:32:48,903 - INFO - Using device: mps
```

## 8.2 Hyperparameter Optimization (PSO)

We used Particle Swarm Optimization (PSO) to find the optimal Learning Rate (LR) and Batch Size (BS). The optimization ran for multiple iterations, testing various combinations.

**Terminal Output - PSO Logs:**
```text
2025-12-07 19:32:48,906 - INFO - === STARTING PSO OPTIMIZATION ===
2025-12-07 19:32:48,907 - INFO - --- PSO Trial: LR=0.008996, BS=54 ---
...
2025-12-07 19:56:37,606 - INFO - PSO Best: LR=0.000163, BS=32
```
*Result:* The optimal hyperparameters found were Learning Rate = 0.000163 and Batch Size = 32.

## 8.3 Model Training

### 8.3.1 Baseline Model (ResNet18)
The baseline model was trained for 100 epochs using standard Cross Entropy Loss.

**Terminal Output - Baseline Training:**
```text
2025-12-07 19:56:37,800 - INFO - --- Starting Baseline Model Training ---
2025-12-07 19:56:42,654 - INFO - Epoch 1 | Val Acc: 0.6458
...
2025-12-07 19:58:31,926 - INFO - Epoch 24 | Val Acc: 0.8594
...
2025-12-07 20:01:21,224 - INFO - Epoch 60 | Val Acc: 0.9115
...
2025-12-07 20:04:31,866 - INFO - Epoch 100 | Val Acc: 0.9010
```

### 8.3.2 MultiTask Model (Proposed Method)
The MultiTask model was trained to predict both Cell Type and Cell Grade simultaneously.

**Terminal Output - MultiTask Training:**
```text
2025-12-07 20:04:33,653 - INFO - --- Starting Multi-Task Model Training ---
2025-12-07 20:04:38,150 - INFO - Epoch 1 | Val Type Acc: 0.6875
...
2025-12-07 20:06:30,883 - INFO - Epoch 25 | Val Type Acc: 0.9531
...
2025-12-07 20:08:40,542 - INFO - Epoch 52 | Val Type Acc: 0.9583
...
2025-12-07 20:12:28,148 - INFO - Epoch 100 | Val Type Acc: 0.9115
```

## 8.4 Quantitative Results

### 8.4.1 Baseline Model Performance
The baseline model achieved an overall accuracy of **92%**.

**Classification Report:**
```text
              precision    recall  f1-score   support

         ALL       1.00      1.00      1.00        48
         AML       0.96      0.94      0.95        48
         CLL       0.85      0.96      0.90        48
         CML       0.88      0.79      0.84        48

    accuracy                           0.92       192
   macro avg       0.92      0.92      0.92       192
weighted avg       0.92      0.92      0.92       192
```

**Confusion Matrix:**
![Baseline Confusion Matrix](/Users/adarshainamdar/Downloads/Major_Project-main/baseline_confusion_matrix.png)

### 8.4.2 MultiTask Model Performance
The proposed MultiTask model achieved an improved overall accuracy of **96%**.

**Classification Report:**
```text
              precision    recall  f1-score   support

         ALL       1.00      1.00      1.00        48
         AML       1.00      1.00      1.00        48
         CLL       0.86      1.00      0.92        48
         CML       1.00      0.83      0.91        48

    accuracy                           0.96       192
   macro avg       0.96      0.96      0.96       192
weighted avg       0.96      0.96      0.96       192
```

**Confusion Matrix:**
![MultiTask Confusion Matrix](/Users/adarshainamdar/Downloads/Major_Project-main/multitask_confusion_matrix.png)

## 8.5 Discussion

The MultiTask Learning approach demonstrated superior performance compared to the Baseline ResNet18 model. 
- **Start-to-End Workflow**: As shown in the terminal outputs, the system successfully loaded raw data, preprocessed it, optimized hyperparameters via PSO, and trained both models.
- **Accuracy Improvement**: The MultiTask model improved accuracy from 92% to 96%.
- **Class-wise Performance**: The MultiTask model achieved perfect precision and recall for Acute Leukemia types (ALL and AML), and significantly improved differentiation between Chronic types (CLL and CML) compared to the baseline.

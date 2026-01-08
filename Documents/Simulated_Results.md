# Simulated Classification Results

Here are the classification results and confusion matrices for the Baseline and MultiTask models.

## 1. Baseline Model (ResNet18)
**Overall Accuracy: 65%**

*   **ALL**: Precision 0.63, Recall 0.73, F1 0.67
*   **AML**: Precision 0.62, Recall 0.54, F1 0.58
*   **CLL**: Precision 0.65, Recall 0.71, F1 0.68
*   **CML**: Precision 0.69, Recall 0.60, F1 0.64

**Detailed Report:**
```text
              precision    recall  f1-score   support

         ALL     0.63      0.73      0.67        48
         AML     0.62      0.54      0.58        48
         CLL     0.65      0.71      0.68        48
         CML     0.69      0.60      0.64        48

    accuracy                         0.65       192
   macro avg     0.65      0.65      0.64       192
weighted avg     0.65      0.65      0.64       192
```

**Confusion Matrix:**
![Baseline Matrix](/Users/adarshainamdar/Downloads/Major_Project-main/baseline_confusion_matrix_simulated.png)

## 2. MultiTask Model (Proposed)
**Overall Accuracy: 89%**

*   **ALL**: Precision 0.93, Recall 0.88, F1 0.90
*   **AML**: Precision 0.90, Recall 0.94, F1 0.92
*   **CLL**: Precision 0.86, Recall 0.88, F1 0.87
*   **CML**: Precision 0.85, Recall 0.85, F1 0.85

**Detailed Report:**
```text
              precision    recall  f1-score   support

         ALL     0.93      0.88      0.90        48
         AML     0.90      0.94      0.92        48
         CLL     0.86      0.88      0.87        48
         CML     0.85      0.85      0.85        48

    accuracy                         0.89       192
   macro avg     0.89      0.89      0.89       192
weighted avg     0.89      0.89      0.89       192
```

**Confusion Matrix:**
![MultiTask Matrix](/Users/adarshainamdar/Downloads/Major_Project-main/multitask_confusion_matrix_simulated.png)

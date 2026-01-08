# LEUKEMIA CLASSIFICATION AND GRADING USING DEEP LEARNING AND PARTICLE SWARM OPTIMIZATION

## ABSTRACT
Acute and classification of leukemia from peripheral blood smear images remains a critical challenge in modern hematopathology. Traditional manual diagnosis is labor-intensive, time-consuming, and subject to inter-observer variability. This project proposes a novel automated framework integrating Deep Learning and Metaheuristic Optimization to classify leukemic cells into four sub-types (ALL, AML, CLL, CML) and simultaneously predict disease severity grades. We employ a Multi-Task ResNet18 architecture, where the feature extraction backbone is shared across classification and grading heads to learn robust representation. To enhance model convergence and generalization, Particle Swarm Optimization (PSO) is utilized to automatically search for optimal hyperparameters, specifically learning rate and batch size. Experimental results on a diverse dataset of microscopic blood images demonstrate that this hybrid approach achieves superior accuracy compared to baseline methods, offering a robust decision support tool for early leukemia detection.

---

## 1. INTRODUCTION

### 1.1 Overview of Hematological Malignancies
Leukemia represents a group of life-threatening malignant disorders of the blood and bone marrow, characterized by the uncontrolled proliferation of abnormal white blood cells. These abnormal cells, often referred to as blasts, interfere with the production of normal blood cells, leading to severe anemia, thrombocytopenia, and immune system failure. According to the World Health Organization (WHO), leukemia is broadly classified based on the lineage of the malignant cells (lymphoid vs. myeloid) and the rate of disease progression (acute vs. chronic). This results in four primary subtypes: Acute Lymphoblastic Leukemia (ALL), Acute Myeloid Leukemia (AML), Chronic Lymphocytic Leukemia (CLL), and Chronic Myeloid Leukemia (CML). Early and precise diagnosis is paramount for selecting appropriate therapeutic regimens, ranging from chemotherapy to stem cell transplantation.

### 1.2 Pathophysiology and Morphological Characteristics
The morphological differentiation of leukemic cells relies on subtle visual cues present in bacterial microscopic examination of peripheral blood smears (PBS) or bone marrow aspirates.
*   **Acute Lymphoblastic Leukemia (ALL)**: Characterized by an overproduction of lymphoblasts. These cells typically exhibit high nuclear-to-cytoplasmic (N/C) ratios, scant cytoplasm, and variably condensed chromatin. The distinguishing of L1, L2, and L3 subtypes (FAB classification) requires expert analysis of nuclear shape and nucleolar prominence.
*   **Acute Myeloid Leukemia (AML)**: Defined by the presence of myeloblasts. Morphologically, these cells are larger than lymphoblasts, possess fine chromatin, and often contain Auer rods—needle-like cytoplasmic inclusions that are pathognomonic for AML.
*   **Chronic Lymphocytic Leukemia (CLL)**: Involves mature-appearing lymphocytes that are biologically incompetent. Smudge cells (ruptured lymphocytes) are a classic artifact seen in PBS.
*   **Chronic Myeloid Leukemia (CML)**: Characterized by the entire spectrum of myeloid differentiation, including neutrophils, myelocytes, metamyelocytes, and bands. The presence of the Philadelphia chromosome ($t(9;22)$) is the genetic hallmark, though morphology remains the initial screening tool.

### 1.3 Challenges in Manual Diagnosis
The standard diagnostic workflow involves the Romanowsky staining of blood films followed by visual inspection under a light microscope. An experienced pathologist must examine hundreds of cells to calculate differential counts. This process is inherently limited by:
1.  **Subjectivity**: Diagnostic concordance between pathologists can vary, especially for ambiguous cell types.
2.  **Fatigue**: The repetitive nature of cell counting leads to observer fatigue, increasing error rates.
3.  **Resource Constraints**: Low-resource settings often lack sufficient hematopathologists, leading to diagnostic delays.
4.  **Biological Heterogeneity**: Leukemic cells exhibit significant intraclass variation in size, texture, and staining characteristics, while different types may share interclass similarities (e.g., distinguishing reactive lymphocytes from blasts).

### 1.4 The Advent of Computational Hematology
Computer-Aided Diagnosis (CAD) has emerged as a transformative solution to these challenges. Early CAD systems relied on traditional image processing techniques such as thresholding, watershed segmentation, and morphological operations to isolate cells. Feature extraction involved manually crafting descriptors for shape (circularity, perimeter), color (histogram moments), and texture (Gray Level Co-occurrence Matrix). These features were then fed into conventional machine learning classifiers like Support Vector Machines (SVM), Random Forests, or Naive Bayes. While effective for small datasets, these systems struggled with generalization due to the rigidity of hand-crafted features.

### 1.5 Deep Learning and Convolutional Neural Networks
The resurgence of artificial intelligence in the last decade, driven by Deep Learning (DL), has revolutionized medical image analysis. Convolutional Neural Networks (CNNs) automate the feature engineering process, learning hierarchical representations directly from raw pixel data. In lower layers, CNNs detect edges and textures, while deeper layers capture abstract concepts like "nuclear irregularity" or "cytoplasmic granularity." Architectures such as AlexNet, VGG, Inception, and ResNet have set new benchmarks in classifying leukocytes.

However, training deep networks requires careful tuning of hyperparameters. The learning rate determines the step size during gradient descent; a value too high leads to divergence, while too low results in slow convergence and entrapment in local minima. Batch size impacts the stability of gradient estimates. Manual tuning (grid search) is computationally expensive and often suboptimal effectively navigating the non-convex loss landscape of deep neural networks.

### 1.6 Metaheuristic Optimization in Deep Learning
To address the optimization challenge, bio-inspired metaheuristic algorithms have been integrated into DL pipelines. Particle Swarm Optimization (PSO), inspired by the social behavior of bird flocking, searches for optimal solutions by moving a population of candidate solutions (particles) through the search space. Each particle adjusts its trajectory based on its own best known position and the swarm's global best position. In the context of CNNs, PSO can efficiently search the hyperparameter space to find configurations that maximize validation accuracy, outperforming random search and grid search in efficiency.

### 1.7 Thesis Contribution and Structure
This project presents an integrated framework for Leukemia classification. We contribute:
1.  A **Multi-Task Learning (MTL) Architecture** based on ResNet18 that simultaneously performs 4-way classification (ALL, AML, CLL, CML) and 3-way grading (Chronic, Accelerated, Blast), leveraging shared feature representations for improved robustness.
2.  A **PSO-driven Hyperparameter Tuning Module** that dynamically optimizes the learning rate and batch size, removing the need for trial-and-error manual tuning.
3.  A comprehensive evaluation on a diverse dataset, demonstrating the efficacy of the proposed hybrid model.

The remainder of this report is organized as follows: Section 2 reviews existing literature. Section 3 details system requirements. Section 4 describes the system design and methodology. Section 5 covers implementation. Section 6 and 7 present testing and results, followed by the conclusion in Section 8.

---

## 2. LITERATURE SURVEY

### 2.1 Literature Review Summary
Recent research (2020-2025) in leukemia diagnosis has shifted heavily towards Deep Learning, with a specific focus on hybrid architectures that combine CNNs with optimization algorithms. Early adoption of basic CNNs (2018-2020) demonstrated the feasibility of automated classification. The subsequent phase (2021-2023) saw the introduction of Transfer Learning using pre-trained weights (ImageNet) to overcome data scarcity in medical domains. The current state-of-the-art (2024-2025) focuses on "Smart" DL models—those incorporating attention mechanisms, Vision Transformers (ViTs), and metaheuristic optimization (PSO, Genetic Algorithms) to enhance precision and explainability. A recurring theme in the literature is the struggle with class imbalance and the need for robust preprocessing to handle staining variations.

### 2.2 Existing Systems
Existing automated systems largely rely on single-task Convolutional Neural Networks. For instance, standard implementations use VGG16 or ResNet50 to classify images into 'Healthy' vs 'Leukemia' or classify specific subtypes. Most of these systems use static hyperparameters fixed by the developer, which may not be optimal for the specific data distribution. Furthermore, few systems attempt to predict the *grade* or phase of the disease, which is clinically distinct from the *type*.

### 2.3 Problem Statement
Despite the high accuracy of modern CNNs, two critical gaps remain. First, **Hyperparameter Sensitivity**: The performance of deep learning models is heavily dependent on hyperparameters (learning rate, batch size), and manual tuning is inefficient. Second, **Clinical Context**: Most models provide a binary or nominal classification without indicating disease severity (grading), which is crucial for treatment planning. This project addresses these by developing a self-tuning, multi-task framework.

### 2.4 Proposed System
We propose a Hybrid Deep Learning system that merges a Multi-Task ResNet18 backbone with Particle Swarm Optimization. The system first utilizes PSO to explore the hyperparameter space and identify the optimal training configuration. It then trains a custom ResNet18 model with dual fully-connected heads: one for classifying the leukemia lineage (ALL/AML/CLL/CML) and another for assessing the grade (Chronic/Accelerated/Blast). This approach ensures both high diagnostic accuracy and clinical relevance.

### 2.5 Objectives
1.  To design a Deep Learning model capable of differentiating four major leukemia types.
2.  To implement a multi-task head for simultaneous disease grading.
3.  To integrate Particle Swarm Optimization for automated hyperparameter tuning.
4.  To develop a user-friendly web interface for real-time inference.

[... Citations Refencing Section ...]

**(References [1]-[20] will be listed in the References section)**

---

## 3. SYSTEM REQUIREMENTS AND SPECIFICATION

### 3.1 Software Requirements
*   **Operating System**: macOS Sequoia / Ubuntu Linux 22.04 LTS / Windows 11
*   **Programming Language**: Python 3.10+
*   **Deep Learning Framework**: PyTorch 2.0+ (torch, torchvision)
*   **Optimization Library**: PySwarm or DEAP
*   **Image Processing**: OpenCV (cv2), Pillow (PIL)
*   **Data Analysis**: NumPy, Pandas, Scikit-learn
*   **Visualization**: Matplotlib, Seaborn
*   **Web Framework**: Flask 3.0
*   **IDE**: VS Code / PyCharm / JupyterLab

### 3.2 Hardware Requirements
*   **Processor**: Apple Silicon M1/M2/M3 or Intel Core i7 (10th Gen+) / AMD Ryzen 7
*   **RAM**: Minimum 16 GB (32 GB recommended for large batch training)
*   **Storage**: 50 GB SSD space (for dataset and model weights)
*   **Display**: 1920x1080 resolution
*   **GPU (Optional but Recommended)**: NVIDIA RTX 3060+ or Apple MPS-enabled Metal GPU for accelerated training.

---

## 4. SYSTEM DESIGN

### 4.1 System Architecture
The system architecture follows a modular pipeline design comprising four distinct stages: Data Engineering, Optimization, Training, and Deployment.

**Stage 1: Data Preprocessing Engine**
Raw images are ingested from the source directory. A preprocessing module applies:
1.  **Normalization**: Scaling pixel values to $[0, 1]$ and normalizing with ImageNet mean $[0.485, 0.456, 0.406]$ and std $[0.229, 0.224, 0.225]$.
2.  **Resizing**: Bicubic interpolation to standard dimension $224 \times 224 \times 3$.
3.  **Augmentation**: During training, stochastic transformations (RandomRotation, HorizontalFlip, ColorJitter) are applied to introduce invariance and prevent overfitting.

**Stage 2: The PSO Optimizer**
Before the main training loop, the PSO engine activates.
*   **Particle Rep**: Each particle represents a vector $[LR, BS]$ (Learning Rate, Batch Size).
*   **Fitness Function**: A "Lite" training run (5 epochs) is executed for each particle position. The validation loss is returned as the fitness cost.
*   **Update Rule**: Particles update their velocities and positions based on local best ($p_{best}$) and global best ($g_{best}$) results.
    $$v_{id}^{t+1} = w \cdot v_{id}^t + c_1 r_1 (p_{best}^t - x_{id}^t) + c_2 r_2 (g_{best}^t - x_{id}^t)$$

**Stage 3: Multi-Task CNN Core**
The core classifier uses ResNet18 connectivity.
*   **Feature Extractor**: Layers `conv1` through `avgpool` are retained from pre-trained ResNet18.
*   **Bifurcation**: The final 512-dimensional embedding vector is fed into two parallel fully connected layers:
    *   $FC_{Type}$: $\mathbb{R}^{512} \rightarrow \mathbb{R}^4$ (Softmax activation for classes)
    *   $FC_{Grade}$: $\mathbb{R}^{512} \rightarrow \mathbb{R}^3$ (Softmax activation for grades)
*   **Loss Function**: A weighted sum of cross-entropy losses:
    $$L_{total} = L_{type} + \lambda \cdot L_{grade}$$ where $\lambda = 0.5$.

**Stage 4: Inference Interface**
A Flask-based web server wraps the trained model. It accepts POST requests with image payloads, preprocesses them, runs the forward pass, and maps distinct logits to readable labels for the UI.


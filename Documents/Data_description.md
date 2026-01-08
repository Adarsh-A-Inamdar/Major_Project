Dataset Description and Statistics
Dataset Description
The dataset is used for a Multi-Task Leukemia Classification and Grading system. It consists of blood cell
images categorized into 4 types of Leukemia, which are further mapped to 2 severity grades.
Classes (Types):
- ALL (Acute Lymphoblastic Leukemia)
- AML (Acute Myeloid Leukemia)
- CLL (Chronic Lymphocytic Leukemia)
- CML (Chronic Myeloid Leukemia)
Grades:
- Acute (Blast): Mapped from ALL and AML
- Chronic: Mapped from CLL and CML
Image Counts
The dataset is balanced with 320 images per class, totaling 1280 images.
It is split into Training (70%), Validation (15%), and Testing (15%).
- Training: 896 images (224 per class)
- Validation: 192 images (48 per class)
- Testing: 192 images (48 per class)
- Total: 1280 images (320 per class)
Size Details
Input Resolution: The system processes all images by resizing them to 224 x 224 pixels.
Channels: 3 (RGB).
Original Formats: The pipeline accepts .jpg, .png, .bmp, and .tif files before resizing them for the model
(ResNet18 backbone).
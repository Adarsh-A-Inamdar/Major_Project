from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Dataset Description and Statistics', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, body)
        self.ln()

pdf = PDF()
pdf.add_page()

# Content Data
desc_text = (
    "The dataset is used for a Multi-Task Leukemia Classification and Grading system. "
    "It consists of blood cell images categorized into 4 types of Leukemia, "
    "which are further mapped to 2 severity grades.\n\n"
    "Classes (Types):\n"
    "  - ALL (Acute Lymphoblastic Leukemia)\n"
    "  - AML (Acute Myeloid Leukemia)\n"
    "  - CLL (Chronic Lymphocytic Leukemia)\n"
    "  - CML (Chronic Myeloid Leukemia)\n\n"
    "Grades:\n"
    "  - Acute (Blast): Mapped from ALL and AML\n"
    "  - Chronic: Mapped from CLL and CML"
)

counts_text = (
    "The dataset is balanced with 320 images per class, totaling 1280 images.\n"
    "It is split into Training (70%), Validation (15%), and Testing (15%).\n\n"
    "  - Training: 896 images (224 per class)\n"
    "  - Validation: 192 images (48 per class)\n"
    "  - Testing: 192 images (48 per class)\n"
    "  - Total: 1280 images (320 per class)"
)

size_text = (
    "Input Resolution: The system processes all images by resizing them to 224 x 224 pixels.\n"
    "Channels: 3 (RGB).\n"
    "Original Formats: The pipeline accepts .jpg, .png, .bmp, and .tif files before resizing them for the model (ResNet18 backbone)."
)

# Write to PDF
pdf.chapter_title('Dataset Description')
pdf.chapter_body(desc_text)

pdf.chapter_title('Image Counts')
pdf.chapter_body(counts_text)

pdf.chapter_title('Size Details')
pdf.chapter_body(size_text)

output_path = "Dataset_Description.pdf"
pdf.output(output_path, 'F')
print(f"PDF generated: {os.path.abspath(output_path)}")

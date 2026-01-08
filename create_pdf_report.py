import os
from PIL import Image
from pathlib import Path

# Setup
FIGURES_DIR = Path('outputs/figures')
OUTPUT_PDF = FIGURES_DIR / 'Research_Graphs_Report.pdf'

def create_pdf():
    images = []
    # List of files in specific order if desired, or just sorted
    # We want them in the order of Figure numbers ideally
    files = sorted(list(FIGURES_DIR.glob('Fig_*.png')))
    
    if not files:
        print("No images found in", FIGURES_DIR)
        return

    print("Found images:")
    for f in files:
        print(f" - {f.name}")
        img = Image.open(f)
        # Convert to RGB just in case (PNG handles RGBA)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        images.append(img)

    if images:
        first_image = images[0]
        rest_images = images[1:]
        
        print(f"\nSaving PDF to {OUTPUT_PDF}...")
        first_image.save(OUTPUT_PDF, "PDF", resolution=100.0, save_all=True, append_images=rest_images)
        print("PDF created successfully!")

if __name__ == "__main__":
    create_pdf()

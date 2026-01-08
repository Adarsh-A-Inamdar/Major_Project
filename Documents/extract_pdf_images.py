from pypdf import PdfReader
from PIL import Image
import io
import os

def extract_images(path):
    print(f"Extracting images from {path}...")
    reader = PdfReader(path)
    output_dir = "outputs/extracted_images"
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for i, page in enumerate(reader.pages):
        for image_file_object in page.images:
            try:
                # The image name in the PDF might not be unique or friendly
                # We'll save it with a sequential name
                count += 1
                image_name = f"extracted_img_{count}_{image_file_object.name}"
                image_path = os.path.join(output_dir, image_name)
                
                with open(image_path, "wb") as fp:
                    fp.write(image_file_object.data)
                
                print(f"Saved {image_path}")
            except Exception as e:
                print(f"Error saving image on page {i}: {e}")

if __name__ == "__main__":
    extract_images("final.pdf")

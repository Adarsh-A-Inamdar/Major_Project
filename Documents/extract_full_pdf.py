from pypdf import PdfReader

def extract_all_text(path):
    print(f"Extracting all text from {path}...")
    try:
        reader = PdfReader(path)
        text = ""
        for i, page in enumerate(reader.pages):
            text += f"\n--- PAGE {i+1} ---\n"
            text += page.extract_text()
        
        # Save to file for easy reading
        with open("final_pdf_content.txt", "w") as f:
            f.write(text)
        print("Text extracted to final_pdf_content.txt")
        
    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    extract_all_text("final.pdf")

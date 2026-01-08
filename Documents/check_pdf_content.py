from pypdf import PdfReader

def analyze_pdf(path):
    print(f"Analyzing {path}...")
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages[:2]: # First 2 pages should be enough
            text += page.extract_text() + "\n"
        
        print("--- Extracted Text Start ---")
        print(text[:2000])
        print("--- Extracted Text End ---")
        
    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    analyze_pdf("final.pdf")

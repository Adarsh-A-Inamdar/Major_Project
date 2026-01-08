from pypdf import PdfReader
import re

def analyze_pdf(path):
    print(f"Analyzing {path}...")
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # Look for headers (rough heuristic: lines that are short, uppercase, or numbered)
        lines = text.split('\n')
        headers = []
        for line in lines:
            line = line.strip()
            # Simple heuristic for headers like "1. Introduction" or "Introduction" in all caps
            if len(line) < 100 and (re.match(r'^\d+\.\s+[A-Z]', line) or line.isupper()):
                headers.append(line)
        
        print("--- Extracted Potential Headers ---")
        seen = set()
        for h in headers:
            if h not in seen:
                print(h)
                seen.add(h)
        print("-----------------------------------")
        
        print(f"Total Pages: {len(reader.pages)}")
        
    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    analyze_pdf("fractalfract-09-00337-v2.pdf")

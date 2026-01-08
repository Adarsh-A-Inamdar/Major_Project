
import markdown
import subprocess
import os
import sys

def convert_md_to_pdf(md_file_path, pdf_file_path):
    print(f"Reading {md_file_path}...")
    with open(md_file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Pre-process image paths to be absolute (cupsfilter might need help or might just work if paths are absolute)
    # The current MD has absolute paths so it should be fine.
    
    # Convert to HTML
    html = markdown.markdown(text, extensions=['extra', 'codehilite'])
    
    # Add minimal CSS for better PDF look
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: "Times New Roman", serif; padding: 40px; line-height: 1.5; }}
            h1, h2, h3 {{ color: #333; }}
            code {{ background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-family: monospace; }}
            pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            .terminal {{ background-color: #333; color: #fff; padding: 10px; border-radius: 5px; font-family: monospace; }}
        </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """
    
    html_file_path = md_file_path.replace('.md', '.html')
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Created temporary HTML: {html_file_path}")

    # Use cupsfilter to convert HTML to PDF
    # usage: cupsfilter -i text/html -m application/pdf input.html > output.pdf
    print("Running cupsfilter...")
    try:
        # We need to capture stdout to file
        with open(pdf_file_path, 'wb') as pdf_file:
            process = subprocess.run(
                ['cupsfilter', '-i', 'text/html', '-m', 'application/pdf', html_file_path],
                check=True,
                stdout=pdf_file,
                stderr=subprocess.PIPE
            )
        print(f"Successfully generated PDF: {pdf_file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running cupsfilter: {e.stderr.decode()}")
    finally:
        # Clean up HTML ? Maybe keep it for debugging
        pass

if __name__ == "__main__":
    md_path = '/Users/adarshainamdar/Downloads/Major_Project-main/Documents/Chapter_8_Results_and_Discussions.md'
    pdf_path = '/Users/adarshainamdar/Downloads/Major_Project-main/Documents/Chapter_8_Results_and_Discussions.pdf'
    convert_md_to_pdf(md_path, pdf_path)

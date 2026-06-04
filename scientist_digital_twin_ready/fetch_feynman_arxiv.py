import os
import time
import requests
from pathlib import Path
import fitz  # pymupdf

arxiv_ids = [
    "2408.15070",  # Feynman 1947 Letter on Path Integral
    "2006.08594",  # Feynman Lectures on the Strong Interactions
    "1902.05799",  # Feynman's Different Approach to Electromagnetism
    "1810.07409",  # The Science and Legacy of Richard Phillips Feynman
    "2311.18410",  # Richard Feynman's Talent for Finding Things Out
    "2209.00083",  # Feynman on Artificial Intelligence
    "1805.03854",  # When Physics Meets Biology - A Less Known Feynman
    "2111.00333",  # A Look Inside Feynman's Route to Gravitation
    "2405.03366",  # Feynman's Simulating Physics with Computers
    "2007.12879",  # Feynman Checkers
    "2210.03365",  # Feynman's Special Perspective on Quantum Mechanics
]

out_dir = Path('data/raw/papers')

session = requests.Session()
session.headers.update({"User-Agent": "ScientistTwin/1.0 (research project)"})

for aid in arxiv_ids:
    pdf_url = f"https://arxiv.org/pdf/{aid}.pdf"
    pdf_path = out_dir / f"arxiv_{aid}.pdf"
    txt_path = out_dir / f"arxiv_{aid}.txt"
    
    if txt_path.exists():
        print(f"Skipping {aid}, already extracted.")
        continue

    print(f"Downloading {aid}...")
    try:
        r = session.get(pdf_url, stream=True, timeout=30)
        if r.status_code == 200:
            with open(pdf_path, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            
            # Extract text
            print(f"Extracting {aid}...")
            doc = fitz.open(str(pdf_path))
            text = '\n'.join(page.get_text() for page in doc)
            doc.close()  # Must close before unlink on Windows
            
            if len(text) > 200:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Saved {txt_path.name} ({len(text)} chars)")
            
            pdf_path.unlink(missing_ok=True)
        else:
            print(f"Failed to download {aid}: HTTP {r.status_code}")
    except Exception as e:
        print(f"Error on {aid}: {e}")
        try:
            pdf_path.unlink(missing_ok=True)
        except:
            pass

    time.sleep(1)

print("All downloads done!")

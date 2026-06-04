import sys, re
files = [
    'src/processing/build_bm25_index.py',
    'src/processing/build_faiss_index.py',
    'src/processing/chunk_store.py',
    'src/processing/chunker.py'
]
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    # Replace open(...) without encoding
    # We will do a generic replacement for with open(...) -> with open(..., encoding='utf-8')
    content = re.sub(r'open\(([^,)]+)\)', r"open(\1, encoding='utf-8')", content)
    content = re.sub(r'open\(([^,)]+), "w"\)', r"open(\1, 'w', encoding='utf-8')", content)
    content = re.sub(r"open\(([^,)]+), 'w'\)", r"open(\1, 'w', encoding='utf-8')", content)
    
    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)
print('Fixed encodings')

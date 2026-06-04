"""
Phase 1.2 - Semantic Scholar Papers & PDF Fetcher.
Fetches all papers, downloads open-access PDFs, extracts text.
No LLM.
"""
import json
import time
import yaml
import requests
from pathlib import Path
from tqdm import tqdm

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
except ImportError:
    pdfminer_extract = None

try:
    import fitz  # pymupdf
except ImportError:
    fitz = None

CONFIG_PATH = Path("config/scientist.yaml")
PAPERS_DIR = Path("data/raw/papers")
MANIFEST_PATH = Path("data/processed/papers/manifest.json")
METADATA_PATH = Path("data/processed/papers/metadata.json")

SS_BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "paperId,title,abstract,year,venue,fieldsOfStudy,openAccessPdf,tldr,citationCount,externalIds"

session = requests.Session()
session.headers.update({
    "User-Agent": "ScientistTwin/1.0 (research project)",
})


def fetch_papers(author_id: str) -> list[dict]:
    papers = []
    offset = 0
    limit = 100
    while True:
        url = f"{SS_BASE}/author/{author_id}/papers"
        params = {"fields": FIELDS, "limit": limit, "offset": offset}
        r = session.get(url, params=params, timeout=10)
        if r.status_code != 200:
            print(f"  SS error {r.status_code} at offset {offset}")
            break
        data = r.json()
        batch = data.get("data", [])
        papers.extend(batch)
        print(f"  Fetched {len(papers)} papers so far...")
        if len(batch) < limit:
            break
        offset += limit
        time.sleep(0.6)
    return papers


def extract_text_from_pdf(pdf_path: Path) -> str:
    text = ""
    if pdfminer_extract:
        try:
            text = pdfminer_extract(str(pdf_path))
        except Exception as e:
            print(f"    pdfminer failed: {e}, trying pymupdf...")
    if not text and fitz:
        try:
            doc = fitz.open(str(pdf_path))
            text = "\n".join(page.get_text() for page in doc)
        except Exception as e:
            print(f"    pymupdf also failed: {e}")
    return text


def clean_text(text: str) -> str:
    import re
    # Remove reference list lines
    text = re.sub(r"^\s*\[\d+\].*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+[A-Z].*$", "", text, flags=re.MULTILINE)
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def download_pdf(url: str, out_path: Path) -> bool:
    try:
        r = session.get(url, timeout=30, stream=True)
        if r.status_code == 200 and "pdf" in r.headers.get("content-type", "").lower():
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"    PDF download error: {e}")
    return False


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    scientist = config["scientist"]
    author_id = str(scientist["semantic_scholar_id"])
    name = scientist["name"]

    print(f"Fetching papers for {name} (SS ID: {author_id})...")
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    papers = fetch_papers(author_id)
    print(f"Total papers returned: {len(papers)}")

    manifest = []
    full_text_count = 0
    abstract_only_count = 0

    for paper in tqdm(papers, desc="Processing papers"):
        pid = paper.get("paperId", "")
        title = paper.get("title", "")
        year = paper.get("year")
        abstract = paper.get("abstract") or ""
        tldr = ""
        if paper.get("tldr"):
            tldr = paper["tldr"].get("text", "")
        fields = paper.get("fieldsOfStudy") or []
        citation_count = paper.get("citationCount", 0)
        oa_pdf = paper.get("openAccessPdf") or {}
        pdf_url = oa_pdf.get("url", "")

        text_path = PAPERS_DIR / f"{pid}.txt"
        has_full_text = False

        if pdf_url and len(abstract) >= 100:
            pdf_path = PAPERS_DIR / f"{pid}.pdf"
            success = download_pdf(pdf_url, pdf_path)
            time.sleep(1.0)
            if success:
                text = extract_text_from_pdf(pdf_path)
                if text and len(text) > 200:
                    cleaned = clean_text(text)
                    text_path.write_text(cleaned, encoding="utf-8")
                    has_full_text = True
                    full_text_count += 1
                    pdf_path.unlink(missing_ok=True)  # save disk space

        if not has_full_text and abstract:
            text_path.write_text(abstract, encoding="utf-8")
            abstract_only_count += 1

        word_count = len(text_path.read_text(encoding="utf-8").split()) if text_path.exists() else 0
        manifest.append({
            "paperId": pid,
            "title": title,
            "year": year,
            "has_full_text": has_full_text,
            "text_path": str(text_path),
            "word_count": word_count,
            "fields": fields,
            "citation_count": citation_count,
            "tldr": tldr,
            "abstract": abstract[:300],
        })

        time.sleep(0.6)

    METADATA_PATH.write_text(json.dumps(papers, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "total_papers": len(papers),
        "papers_with_full_text": full_text_count,
        "papers_abstract_only": abstract_only_count,
        "papers": manifest,
    }
    MANIFEST_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✅ Done. Total: {len(papers)} | Full text: {full_text_count} | Abstract only: {abstract_only_count}")
    print(f"Manifest saved to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

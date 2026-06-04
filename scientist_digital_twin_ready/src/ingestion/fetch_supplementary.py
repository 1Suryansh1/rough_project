"""
Phase 1.5 - Nobel Lecture, arXiv, and Open-Access Fetcher.
Three tasks in one script. No LLM.
"""
import json
import re
import time
import xml.etree.ElementTree as ET
import yaml
import requests
from html.parser import HTMLParser
from pathlib import Path

CONFIG_PATH = Path("config/scientist.yaml")
PAPERS_DIR = Path("data/raw/papers")
TRANSCRIPTS_DIR = Path("data/raw/transcripts")
MANIFEST_PATH = Path("data/processed/papers/manifest.json")

session = requests.Session()
session.headers.update({"User-Agent": "ScientistTwin/1.0 (research project)"})


class TextExtractor(HTMLParser):
    SKIP_TAGS = {"script", "style", "nav", "header", "footer"}

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.SKIP_TAGS:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag.lower() in self.SKIP_TAGS:
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data):
        if self._skip == 0:
            stripped = data.strip()
            if stripped:
                self.text_parts.append(stripped)

    def get_text(self) -> str:
        return "\n".join(self.text_parts)


def fetch_nobel_lecture(url: str) -> str:
    print(f"Fetching Nobel lecture: {url}")
    r = session.get(url, timeout=15)
    r.raise_for_status()
    parser = TextExtractor()
    parser.feed(r.text)
    text = parser.get_text()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_overlap_ratio(a: str, b: str) -> float:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def fetch_arxiv(query: str) -> list[dict]:
    url = "http://export.arxiv.org/api/query"
    params = {"search_query": f"au:{query}", "max_results": 50, "sortBy": "submittedDate"}
    time.sleep(3)
    r = session.get(url, params=params, timeout=20)
    r.raise_for_status()
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)
    papers = []
    for entry in root.findall("atom:entry", ns):
        arxiv_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
        title = entry.find("atom:title", ns).text.strip()
        abstract = entry.find("atom:summary", ns).text.strip()
        published = entry.find("atom:published", ns).text[:10]
        cats = [c.get("term") for c in entry.findall("atom:category", ns)]
        papers.append({"arxiv_id": arxiv_id, "title": title, "abstract": abstract,
                        "published": published, "categories": cats})
    return papers


def download_arxiv_pdf(arxiv_id: str, out_path: Path) -> bool:
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        r = session.get(pdf_url, timeout=30, stream=True)
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"    arxiv PDF error: {e}")
    return False


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    scientist = config["scientist"]

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    # TASK A: Nobel Lecture
    nobel_url = scientist.get("nobel_lecture_url", "")
    if nobel_url:
        try:
            text = fetch_nobel_lecture(nobel_url)
            out = TRANSCRIPTS_DIR / "nobel_lecture.txt"
            out.write_text(text, encoding="utf-8")
            print(f"✅ Nobel lecture: {len(text.split())} words → {out}")
        except Exception as e:
            print(f"⚠️  Nobel lecture failed: {e}")

    # TASK B: arXiv
    arxiv_query = scientist.get("arxiv_author_query", "")
    if arxiv_query:
        print(f"\nFetching arXiv papers for query: {arxiv_query}")
        try:
            arxiv_papers = fetch_arxiv(arxiv_query)
            # Load existing manifest to deduplicate
            existing_titles = []
            if MANIFEST_PATH.exists():
                with open(MANIFEST_PATH) as f:
                    manifest = json.load(f)
                existing_titles = [p.get("title", "") for p in manifest.get("papers", [])]
            new_count = 0
            for paper in arxiv_papers:
                if any(word_overlap_ratio(paper["title"], t) > 0.85 for t in existing_titles):
                    continue
                arxiv_id_clean = paper["arxiv_id"].replace("/", "_")
                txt_path = PAPERS_DIR / f"arxiv_{arxiv_id_clean}.txt"
                txt_path.write_text(paper["abstract"], encoding="utf-8")
                existing_titles.append(paper["title"])
                new_count += 1
                time.sleep(3)
            print(f"✅ arXiv: {new_count} new papers added")
        except Exception as e:
            print(f"⚠️  arXiv fetch failed: {e}")

    # TASK C: Open-access URLs from config
    oa_urls = scientist.get("open_access_urls", [])
    for entry in oa_urls:
        url = entry if isinstance(entry, str) else entry.get("url", "")
        slug = entry.get("slug", "oa_doc") if isinstance(entry, dict) else "oa_doc"
        try:
            r = session.get(url, timeout=15)
            parser = TextExtractor()
            parser.feed(r.text)
            text = parser.get_text()
            out = TRANSCRIPTS_DIR / f"{slug}.txt"
            out.write_text(text, encoding="utf-8")
            print(f"✅ OA doc {slug}: {len(text.split())} words")
        except Exception as e:
            print(f"⚠️  OA doc {slug} failed: {e}")


if __name__ == "__main__":
    main()

"""
Phase 2.1 - Section-Aware Semantic Chunker.
Handles papers, transcripts, and Wikipedia with different strategies.
No LLM.
"""
import re
import json
import uuid
from pathlib import Path
from typing import Generator
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

CHUNKS_PATH = Path("data/processed/all_chunks.json")
REPORT_PATH = Path("data/processed/chunking_report.json")
PAPERS_DIR = Path("data/raw/papers")
TRANSCRIPTS_DIR = Path("data/raw/transcripts")
MANIFEST_PATH = Path("data/processed/papers/manifest.json")

MIN_WORDS = 80
MAX_WORDS = 500
TARGET_WORDS_PAPER = 300
TARGET_WORDS_TRANSCRIPT = 250

SEMANTIC_THRESHOLD = 0.35

# Lazy-loaded — only for boundary detection
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("Loading all-MiniLM-L6-v2 for boundary detection...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def semantic_sub_chunk(text: str, target_words: int) -> list[str]:
    sentences = split_into_sentences(text)
    if len(sentences) <= 2:
        return [text]
    model = get_model()
    embeddings = model.encode(sentences, show_progress_bar=False)
    # find breakpoints where similarity drops
    breaks = []
    for i in range(len(sentences) - 1):
        sim = cosine_sim(embeddings[i], embeddings[i + 1])
        if sim < SEMANTIC_THRESHOLD:
            breaks.append(i + 1)
    # build segments from breaks
    segments = []
    prev = 0
    for b in breaks:
        segments.append(" ".join(sentences[prev:b]))
        prev = b
    segments.append(" ".join(sentences[prev:]))
    # merge short segments
    merged = []
    buf = ""
    for seg in segments:
        if not buf:
            buf = seg
        elif len((buf + " " + seg).split()) < target_words:
            buf = buf + " " + seg
        else:
            merged.append(buf)
            buf = seg
    if buf:
        merged.append(buf)
    return merged or [text]


PAPER_SECTION_PATTERN = re.compile(
    r"^(?:[A-Z][A-Z\s]{2,}|(?:\d+\.?\d*)\s+[A-Z][a-zA-Z\s]+)$",
    re.MULTILINE,
)


def detect_paper_sections(text: str) -> list[tuple[str, str]]:
    """Returns list of (section_title, section_text)."""
    lines = text.split("\n")
    sections = []
    current_title = "Body"
    current_lines = []
    section_keywords = {
        "abstract", "introduction", "background", "methods", "methodology",
        "results", "discussion", "conclusion", "references", "acknowledgements",
        "related work", "experiments", "evaluation", "appendix",
    }
    for line in lines:
        stripped = line.strip()
        is_heading = (
            (stripped.isupper() and 2 < len(stripped) < 60)
            or re.match(r"^\d+\.?\d*\s+[A-Z]", stripped)
            or stripped.lower() in section_keywords
        )
        if is_heading and len(stripped) < 80:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = stripped
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return sections


def chunk_paper(text: str, metadata_base: dict) -> list[dict]:
    chunks = []
    sections = detect_paper_sections(text)
    for section_title, section_text in sections:
        word_count = len(section_text.split())
        if word_count < MIN_WORDS:
            continue
        if word_count <= MAX_WORDS:
            sub_chunks = [section_text]
        else:
            sub_chunks = semantic_sub_chunk(section_text, TARGET_WORDS_PAPER)
        for i, sub in enumerate(sub_chunks):
            wc = len(sub.split())
            if wc < MIN_WORDS:
                continue
            if wc > MAX_WORDS:
                # hard split
                words = sub.split()
                sub = " ".join(words[:MAX_WORDS])
            meta = {**metadata_base, "section": section_title, "chunk_idx": i}
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": sub.strip(),
                "word_count": len(sub.split()),
                "metadata": meta,
            })
    return chunks


def chunk_transcript(text: str, metadata_base: dict) -> list[dict]:
    chunks = []
    # Split on double newline or SECTION markers
    paragraphs = re.split(r"\n{2,}|(?=\[?\d{2}:\d{2}:\d{2}\]?)", text)
    timestamp_re = re.compile(r"\[?(\d{2}:\d{2}:\d{2})\]?")

    buf = ""
    buf_timestamp = None
    chunk_idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        ts_match = timestamp_re.match(para)
        if ts_match:
            buf_timestamp = ts_match.group(1)
            para = para[ts_match.end():].strip()

        wc_para = len(para.split())
        if wc_para < 50:
            buf = (buf + " " + para).strip() if buf else para
            continue

        if len((buf + " " + para).split()) >= TARGET_WORDS_TRANSCRIPT:
            if buf and len(buf.split()) >= MIN_WORDS:
                meta = {**metadata_base, "chunk_idx": chunk_idx, "timestamp_start": buf_timestamp}
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "text": buf.strip(),
                    "word_count": len(buf.split()),
                    "metadata": meta,
                })
                chunk_idx += 1
            buf = para
        else:
            buf = (buf + " " + para).strip() if buf else para

    if buf and len(buf.split()) >= MIN_WORDS:
        meta = {**metadata_base, "chunk_idx": chunk_idx, "timestamp_start": buf_timestamp}
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "text": buf.strip(),
            "word_count": len(buf.split()),
            "metadata": meta,
        })

    # Handle large paragraphs
    final = []
    for chunk in chunks:
        if chunk["word_count"] > MAX_WORDS:
            subs = semantic_sub_chunk(chunk["text"], TARGET_WORDS_TRANSCRIPT)
            for i, sub in enumerate(subs):
                wc = len(sub.split())
                if wc < MIN_WORDS:
                    continue
                meta = {**chunk["metadata"], "chunk_idx": chunk["metadata"]["chunk_idx"] * 100 + i}
                final.append({
                    "chunk_id": str(uuid.uuid4()),
                    "text": sub.strip(),
                    "word_count": wc,
                    "metadata": meta,
                })
        else:
            final.append(chunk)
    return final


def main():
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    # --- Papers ---
    manifest = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding='utf-8') as f:
            manifest_data = json.load(f)
        for p in manifest_data.get("papers", []):
            manifest[p["paperId"]] = p

    print("\n[1/3] Chunking papers...")
    paper_count = 0
    for txt_path in tqdm(list(PAPERS_DIR.glob("*.txt"))):
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        pid = txt_path.stem
        paper_meta = manifest.get(pid, {})
        meta_base = {
            "source": "paper",
            "paper_id": pid,
            "title": paper_meta.get("title", ""),
            "year": paper_meta.get("year"),
            "fields": paper_meta.get("fields", []),
        }
        chunks = chunk_paper(text, meta_base)
        all_chunks.extend(chunks)
        paper_count += len(chunks)

    # --- Transcripts & Wikipedia ---
    print("\n[2/3] Chunking transcripts & Wikipedia...")
    transcript_count = 0
    for txt_path in tqdm(list(TRANSCRIPTS_DIR.glob("*.txt"))):
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        name = txt_path.stem
        meta_base = {"source": "transcript", "video_id_or_name": name, "year": None}
        chunks = chunk_transcript(text, meta_base)
        all_chunks.extend(chunks)
        transcript_count += len(chunks)

    print(f"\n[3/3] Saving {len(all_chunks)} total chunks...")
    CHUNKS_PATH.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    word_counts = [c["word_count"] for c in all_chunks]
    report = {
        "total_chunks": len(all_chunks),
        "by_source": {"papers": paper_count, "transcripts": transcript_count},
        "avg_word_count": round(sum(word_counts) / len(word_counts), 1) if word_counts else 0,
        "min_word_count": min(word_counts) if word_counts else 0,
        "max_word_count": max(word_counts) if word_counts else 0,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"✅ Chunks saved → {CHUNKS_PATH}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

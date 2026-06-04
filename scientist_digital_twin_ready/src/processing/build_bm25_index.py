"""
Phase 2.2 - BM25s Index Builder.
Builds BM25 index over all chunks. No LLM.
"""
import json
import re
from pathlib import Path
from tqdm import tqdm
import bm25s

CHUNKS_PATH = Path("data/processed/all_chunks.json")
INDEX_DIR = Path("data/indices")
BM25_INDEX_PATH = INDEX_DIR / "bm25_index"
BM25_CHUNK_IDS_PATH = INDEX_DIR / "bm25_chunk_ids.json"


def tokenize(text: str) -> list[str]:
    """Consistent tokenizer — used for both indexing and retrieval."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def build_index() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading chunks...")
    with open(CHUNKS_PATH, encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"Tokenizing {len(chunks)} chunks...")
    tokenized_corpus = [tokenize(c["text"]) for c in tqdm(chunks)]
    chunk_ids = [c["chunk_id"] for c in chunks]

    print("Building BM25s index...")
    retriever = bm25s.BM25()
    retriever.index(tokenized_corpus)
    retriever.save(str(BM25_INDEX_PATH))
    BM25_CHUNK_IDS_PATH.write_text(json.dumps(chunk_ids), encoding="utf-8")
    print(f"✅ BM25 index saved → {BM25_INDEX_PATH}")
    print(f"✅ Chunk ID map → {BM25_CHUNK_IDS_PATH}")


def bm25_retrieve(query: str, top_k: int = 20) -> list[dict]:
    retriever = bm25s.BM25.load(str(BM25_INDEX_PATH))
    with open(BM25_CHUNK_IDS_PATH, encoding='utf-8') as f:
        chunk_ids = json.load(f)
    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []
    results, scores = retriever.retrieve([tokenized_query], k=min(top_k, len(chunk_ids)))
    output = []
    for idx, score in zip(results[0], scores[0]):
        output.append({"chunk_id": chunk_ids[idx], "bm25_rank": len(output) + 1, "bm25_score": float(score)})
    return output


if __name__ == "__main__":
    build_index()
    # Self-test
    test_queries = [
        "path integral formulation quantum mechanics",
        "Feynman diagrams perturbation theory",
        "Nobel Prize physics 1965",
    ]
    for q in test_queries:
        results = bm25_retrieve(q, top_k=3)
        print(f"\nQuery: '{q}'")
        for r in results:
            print(f"  chunk_id={r['chunk_id'][:8]}... score={r['bm25_score']:.3f}")

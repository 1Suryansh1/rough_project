"""
Phase 2.3 - FAISS HNSW Dual-Encoder Index Builder.
SPECTER2 for papers, all-mpnet-base-v2 for transcripts/Wikipedia.
No LLM.
"""
import json
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/processed/all_chunks.json")
INDEX_DIR = Path("data/indices")
FAISS_INDEX_PATH = INDEX_DIR / "faiss_hnsw.index"
FAISS_CHUNK_IDS_PATH = INDEX_DIR / "faiss_chunk_ids.json"
EMBEDDING_MANIFEST_PATH = INDEX_DIR / "embedding_manifest.json"
EMBEDDING_MODELS_PATH = INDEX_DIR / "embedding_models.json"

SPECTER_MODEL = "allenai/specter"   # allenai/specter2_base requires the `adapters` library;
                                     # allenai/specter is natively packaged as a SentenceTransformer
MPNET_MODEL = "all-mpnet-base-v2"
DIM = 768


def encode_batch(model: SentenceTransformer, texts: list[str], batch_size: int = 32) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def build_index() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading chunks...")
    with open(CHUNKS_PATH, encoding='utf-8') as f:
        chunks = json.load(f)

    paper_chunks = [c for c in chunks if c["metadata"].get("source") == "paper"]
    other_chunks = [c for c in chunks if c["metadata"].get("source") != "paper"]
    print(f"Paper chunks: {len(paper_chunks)} | Other chunks: {len(other_chunks)}")

    all_embeddings = []
    all_chunk_ids = []
    embedding_manifest = []

    # Encode papers with SPECTER2
    if paper_chunks:
        print(f"\nEncoding {len(paper_chunks)} paper chunks with SPECTER...")
        specter = SentenceTransformer(SPECTER_MODEL)
        paper_texts = [c["text"] for c in paper_chunks]
        paper_embs = encode_batch(specter, paper_texts)
        all_embeddings.append(paper_embs)
        all_chunk_ids.extend(c["chunk_id"] for c in paper_chunks)
        for i, c in enumerate(paper_chunks):
            embedding_manifest.append({
                "chunk_id": c["chunk_id"],
                "faiss_idx": i,
                "model": "specter",
            })
        del specter

    # Encode others with mpnet
    if other_chunks:
        print(f"\nEncoding {len(other_chunks)} transcript/wiki chunks with all-mpnet-base-v2...")
        mpnet = SentenceTransformer(MPNET_MODEL)
        other_texts = [c["text"] for c in other_chunks]
        other_embs = encode_batch(mpnet, other_texts)
        offset = len(paper_chunks)
        for i, c in enumerate(other_chunks):
            embedding_manifest.append({
                "chunk_id": c["chunk_id"],
                "faiss_idx": offset + i,
                "model": "mpnet",
            })
        all_embeddings.append(other_embs)
        all_chunk_ids.extend(c["chunk_id"] for c in other_chunks)
        del mpnet

    all_embs = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    print(f"\nTotal embeddings: {all_embs.shape}")

    # Build FAISS HNSW
    print("Building FAISS HNSW index...")
    index = faiss.IndexHNSWFlat(DIM, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64
    index.add(all_embs)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    FAISS_CHUNK_IDS_PATH.write_text(json.dumps(all_chunk_ids), encoding="utf-8")
    EMBEDDING_MANIFEST_PATH.write_text(
        json.dumps(embedding_manifest, indent=2), encoding="utf-8"
    )
    EMBEDDING_MODELS_PATH.write_text(
        json.dumps({
            "specter_model": SPECTER_MODEL,
            "mpnet_model": MPNET_MODEL,
            "dim": DIM,
            "normalize": True,
            "total_vectors": int(index.ntotal),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"✅ FAISS index ({index.ntotal} vectors) → {FAISS_INDEX_PATH}")


def faiss_retrieve(query: str, source_type: str = "auto", top_k: int = 20) -> list[dict]:
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(FAISS_CHUNK_IDS_PATH, encoding='utf-8') as f:
        chunk_ids = json.load(f)
    with open(EMBEDDING_MODELS_PATH, encoding='utf-8') as f:
        models_info = json.load(f)

    if source_type == "paper":
        model = SentenceTransformer(models_info["specter_model"])
    elif source_type == "other":
        model = SentenceTransformer(models_info["mpnet_model"])
    else:  # auto — encode with both and average
        specter = SentenceTransformer(models_info["specter_model"])
        mpnet = SentenceTransformer(models_info["mpnet_model"])
        e1 = specter.encode([query], normalize_embeddings=True).astype(np.float32)
        e2 = mpnet.encode([query], normalize_embeddings=True).astype(np.float32)
        query_vec = ((e1 + e2) / 2).astype(np.float32)
        faiss.normalize_L2(query_vec)
        scores, indices = index.search(query_vec, top_k)
        return [
            {"chunk_id": chunk_ids[idx], "faiss_rank": rank + 1, "faiss_score": float(score)}
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0]))
            if idx >= 0
        ]

    query_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(query_vec, top_k)
    return [
        {"chunk_id": chunk_ids[idx], "faiss_rank": rank + 1, "faiss_score": float(score)}
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]))
        if idx >= 0
    ]


if __name__ == "__main__":
    build_index()

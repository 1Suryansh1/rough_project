"""
Phase 4.1 - RRF Retrieval Engine.
BM25 + FAISS with Reciprocal Rank Fusion. No LLM.
"""
import json
import re
import yaml
import sqlite3
from pathlib import Path
from typing import Optional
import numpy as np
import faiss
import bm25s
from sentence_transformers import SentenceTransformer

CONFIG_PATH = Path("config/scientist.yaml")
INDEX_DIR = Path("data/indices")
DB_PATH = Path("data/processed/chunks.db")
WIKIDATA_PATH = Path("data/processed/scientist_wikidata.json")

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "what",
    "how", "why", "when", "where", "which", "who", "and", "or",
    "but", "in", "on", "at", "to", "for", "of", "with", "about",
}


class RetrievalEngine:
    def __init__(self, config_path: Path = CONFIG_PATH):
        with open(config_path) as f:
            self._config = yaml.safe_load(f)

        # Load BM25
        self._bm25 = bm25s.BM25.load(str(INDEX_DIR / "bm25_index"))
        with open(INDEX_DIR / "bm25_chunk_ids.json") as f:
            self._bm25_chunk_ids: list[str] = json.load(f)

        # Load FAISS
        self._faiss_index = faiss.read_index(str(INDEX_DIR / "faiss_hnsw.index"))
        with open(INDEX_DIR / "faiss_chunk_ids.json") as f:
            self._faiss_chunk_ids: list[str] = json.load(f)

        with open(INDEX_DIR / "embedding_models.json") as f:
            self._models_info = json.load(f)

        # Load Wikidata
        with open(WIKIDATA_PATH, "r", encoding="utf-8") as f:
            self._wikidata = json.load(f)

        # Load signature vocab for query routing
        profile_path = Path("data/processed/scientist_profile.json")
        self._sig_vocab: set[str] = set()
        if profile_path.exists():
            with open(profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
            self._sig_vocab = set(v.lower() for v in profile.get("signature_vocabulary", []))

        # Lazy-loaded models
        self._specter2: Optional[SentenceTransformer] = None
        self._mpnet: Optional[SentenceTransformer] = None

    def _get_specter(self) -> SentenceTransformer:
        if self._specter2 is None:
            self._specter2 = SentenceTransformer(self._models_info["specter_model"])
        return self._specter2

    def _get_mpnet(self) -> SentenceTransformer:
        if self._mpnet is None:
            self._mpnet = SentenceTransformer(self._models_info["mpnet_model"])
        return self._mpnet

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s-]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    def bm25_retrieve(self, query: str, top_k: int = 30) -> list[dict]:
        tokens = self._tokenize(query)
        if not tokens:
            return []
        results, scores = self._bm25.retrieve([tokens], k=min(top_k, len(self._bm25_chunk_ids)))
        return [
            {"chunk_id": self._bm25_chunk_ids[idx], "bm25_rank": rank + 1, "bm25_score": float(score)}
            for rank, (idx, score) in enumerate(zip(results[0], scores[0]))
        ]

    def _query_has_scientific_terms(self, query: str) -> bool:
        tokens = set(query.lower().split())
        return bool(tokens & self._sig_vocab)

    def faiss_retrieve(self, query: str, top_k: int = 30) -> list[dict]:
        if self._query_has_scientific_terms(query):
            # Use both models, average embeddings
            e1 = self._get_specter().encode([query], normalize_embeddings=True).astype(np.float32)
            e2 = self._get_mpnet().encode([query], normalize_embeddings=True).astype(np.float32)
            query_vec = ((e1 + e2) / 2).astype(np.float32)
            faiss.normalize_L2(query_vec)
        else:
            query_vec = self._get_mpnet().encode([query], normalize_embeddings=True).astype(np.float32)

        scores, indices = self._faiss_index.search(query_vec, top_k)
        return [
            {"chunk_id": self._faiss_chunk_ids[idx], "faiss_rank": rank + 1, "faiss_score": float(score)}
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0]))
            if idx >= 0
        ]

    @staticmethod
    def rrf_fuse(bm25_results: list[dict], faiss_results: list[dict], k: int = 60) -> list[dict]:
        scores: dict[str, float] = {}
        in_bm25: set[str] = set()
        in_faiss: set[str] = set()

        for item in bm25_results:
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + item["bm25_rank"])
            in_bm25.add(cid)
        for item in faiss_results:
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + item["faiss_rank"])
            in_faiss.add(cid)

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [
            {
                "chunk_id": cid,
                "rrf_score": scores[cid],
                "in_bm25": cid in in_bm25,
                "in_faiss": cid in in_faiss,
            }
            for cid in sorted_ids
        ]

    def get_wikidata_context(self, query: str) -> str:
        tokens = [t for t in query.lower().split() if t not in STOP_WORDS and len(t) > 3]
        name = self._wikidata.get("name", "")
        lines = [
            "STRUCTURED FACTS",
            f"- name: {name}",
        ]
        always_keys = ["fields_of_work", "date_of_birth", "date_of_death", "awards_received",
                       "notable_works", "employers", "doctoral_advisor"]
        for key in always_keys:
            val = self._wikidata.get(key)
            if val:
                if isinstance(val, list):
                    display = ", ".join(v["value"] if isinstance(v, dict) else str(v) for v in val[:5])
                else:
                    display = str(val)
                lines.append(f"- {key.replace('_', ' ')}: {display}")

        # Keyword match for relevant facts
        for key, val in self._wikidata.items():
            if key in always_keys or key in {"name", "qid", "description", "gender"}:
                continue
            val_str = json.dumps(val).lower()
            if any(tok in val_str for tok in tokens):
                if isinstance(val, list):
                    display = ", ".join(v["value"] if isinstance(v, dict) else str(v) for v in val[:5])
                else:
                    display = str(val)
                lines.append(f"- {key.replace('_', ' ')}: {display}")

        return "\n".join(lines)

    def retrieve(self, query: str, top_k: int = 8) -> dict:
        bm25_results = self.bm25_retrieve(query, 30)
        faiss_results = self.faiss_retrieve(query, 30)
        fused = self.rrf_fuse(bm25_results, faiss_results)
        top_fused = fused[:top_k]

        # Fetch chunks from DB
        chunk_ids = [r["chunk_id"] for r in top_fused]
        conn = sqlite3.connect(str(DB_PATH))
        placeholders = ",".join("?" * len(chunk_ids))
        rows = conn.execute(
            f"SELECT chunk_id, text, word_count, source, paper_id, section, year, metadata_json "
            f"FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        conn.close()

        id_to_chunk = {r[0]: {
            "chunk_id": r[0], "text": r[1], "word_count": r[2],
            "source": r[3], "paper_id": r[4], "section": r[5],
            "year": r[6], "metadata": json.loads(r[7] or "{}"),
        } for r in rows}
        chunks = [id_to_chunk[cid] for cid in chunk_ids if cid in id_to_chunk]

        wikidata_context = self.get_wikidata_context(query)
        bm25_cids = {r["chunk_id"] for r in bm25_results[:top_k]}
        faiss_cids = {r["chunk_id"] for r in faiss_results[:top_k]}
        top_cids = set(chunk_ids)
        both = top_cids & bm25_cids & faiss_cids

        return {
            "chunks": chunks,
            "wikidata_context": wikidata_context,
            "bm25_count": len(top_cids & bm25_cids),
            "faiss_count": len(top_cids & faiss_cids),
            "both_count": len(both),
        }


if __name__ == "__main__":
    engine = RetrievalEngine()
    test_queries = [
        "What is the path integral formulation?",
        "How did Feynman approach teaching physics?",
        "What is quantum electrodynamics?",
        "Who were Feynman's doctoral students?",
        "What did Feynman think about the Copenhagen interpretation?",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        result = engine.retrieve(q, top_k=3)
        print(f"  BM25: {result['bm25_count']} | FAISS: {result['faiss_count']} | Both: {result['both_count']}")
        for i, c in enumerate(result["chunks"]):
            print(f"  [{i+1}] ({c['source']}) {c['text'][:120]}...")
        print(f"  Wikidata context ({len(result['wikidata_context'])} chars)")

"""
Phase 2.4 - SQLite Chunk Store.
Master record for all chunks + Wikidata facts. No LLM.
"""
import json
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path("data/processed/chunks.db")
CHUNKS_PATH = Path("data/processed/all_chunks.json")
BM25_IDS_PATH = Path("data/indices/bm25_chunk_ids.json")
FAISS_IDS_PATH = Path("data/indices/faiss_chunk_ids.json")
WIKIDATA_PATH = Path("data/processed/scientist_wikidata.json")


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            word_count INTEGER,
            source TEXT,
            paper_id TEXT,
            section TEXT,
            year INTEGER,
            bm25_doc_idx INTEGER,
            faiss_idx INTEGER,
            embedding_model TEXT,
            metadata_json TEXT
        );
        CREATE TABLE IF NOT EXISTS wikidata_facts (
            fact_key TEXT PRIMARY KEY,
            fact_value TEXT,
            source_qid TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
        CREATE INDEX IF NOT EXISTS idx_chunks_year ON chunks(year);
    """)
    conn.commit()
    return conn


def insert_chunks(
    chunks: list[dict],
    bm25_ids: dict[str, int],
    faiss_ids: dict[str, int],
    embedding_manifest: dict[str, str],
    db_path: Path = DB_PATH,
) -> None:
    conn = init_db(db_path)
    rows = []
    for c in chunks:
        meta = c.get("metadata", {})
        cid = c["chunk_id"]
        rows.append((
            cid,
            c["text"],
            c.get("word_count", len(c["text"].split())),
            meta.get("source"),
            meta.get("paper_id"),
            meta.get("section"),
            meta.get("year"),
            bm25_ids.get(cid),
            faiss_ids.get(cid),
            embedding_manifest.get(cid),
            json.dumps(meta),
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    print(f"✅ Inserted {len(rows)} chunks into DB")


def get_chunks_by_ids(chunk_ids: list[str], db_path: Path = DB_PATH) -> list[dict]:
    conn = init_db(db_path)
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT chunk_id, text, word_count, source, paper_id, section, year, metadata_json "
        f"FROM chunks WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    conn.close()
    return [
        {
            "chunk_id": r[0], "text": r[1], "word_count": r[2],
            "source": r[3], "paper_id": r[4], "section": r[5],
            "year": r[6], "metadata": json.loads(r[7] or "{}"),
        }
        for r in rows
    ]


def insert_wikidata_facts(wikidata_dict: dict, db_path: Path = DB_PATH) -> None:
    conn = init_db(db_path)
    qid = wikidata_dict.get("qid", "")
    rows = []
    for key, value in wikidata_dict.items():
        rows.append((key, json.dumps(value, ensure_ascii=False), qid))
    conn.executemany(
        "INSERT OR REPLACE INTO wikidata_facts VALUES (?,?,?)", rows
    )
    conn.commit()
    conn.close()
    print(f"✅ Inserted {len(rows)} Wikidata facts into DB")


def get_wikidata_context(query_entities: list[str], db_path: Path = DB_PATH) -> str:
    conn = init_db(db_path)
    always_include = ["name", "fields_of_work", "date_of_birth", "date_of_death",
                      "awards_received", "notable_works", "doctoral_advisor"]
    all_keys = always_include.copy()
    for entity in query_entities:
        rows = conn.execute(
            "SELECT fact_key FROM wikidata_facts WHERE LOWER(fact_value) LIKE ?",
            (f"%{entity.lower()}%",)
        ).fetchall()
        all_keys.extend(r[0] for r in rows)

    all_keys = list(dict.fromkeys(all_keys))  # deduplicate preserving order
    lines = ["STRUCTURED FACTS"]
    for key in all_keys:
        row = conn.execute(
            "SELECT fact_value FROM wikidata_facts WHERE fact_key=?", (key,)
        ).fetchone()
        if row:
            val = json.loads(row[0])
            if isinstance(val, list):
                display = ", ".join(
                    v["value"] if isinstance(v, dict) else str(v) for v in val[:5]
                )
            else:
                display = str(val)
            lines.append(f"- {key.replace('_', ' ')}: {display}")
    conn.close()
    return "\n".join(lines)


def populate_db() -> None:
    print("Loading all_chunks.json...")
    with open(CHUNKS_PATH, encoding='utf-8') as f:
        chunks = json.load(f)

    print("Loading BM25 chunk IDs...")
    with open(BM25_IDS_PATH, encoding='utf-8') as f:
        bm25_list = json.load(f)
    bm25_ids = {cid: i for i, cid in enumerate(bm25_list)}

    print("Loading FAISS chunk IDs...")
    with open(FAISS_IDS_PATH, encoding='utf-8') as f:
        faiss_list = json.load(f)
    faiss_ids = {cid: i for i, cid in enumerate(faiss_list)}

    # embedding model per chunk
    manifest_path = Path("data/indices/embedding_manifest.json")
    embedding_manifest: dict[str, str] = {}
    if manifest_path.exists():
        with open(manifest_path, encoding='utf-8') as f:
            mdata = json.load(f)
        embedding_manifest = {m["chunk_id"]: m["model"] for m in mdata}

    insert_chunks(chunks, bm25_ids, faiss_ids, embedding_manifest)

    print("Loading Wikidata JSON...")
    if WIKIDATA_PATH.exists():
        with open(WIKIDATA_PATH, encoding='utf-8') as f:
            wikidata = json.load(f)
        insert_wikidata_facts(wikidata)

    conn = sqlite3.connect(str(DB_PATH))
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    n_facts = conn.execute("SELECT COUNT(*) FROM wikidata_facts").fetchone()[0]
    conn.close()
    print(f"\n✅ DB populated: {n_chunks} chunks, {n_facts} Wikidata facts")


if __name__ == "__main__":
    populate_db()

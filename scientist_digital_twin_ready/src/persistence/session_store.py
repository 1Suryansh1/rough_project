"""
Phase 5.2 - Session Store (SQLite).
All session/turn persistence. No LLM, no external services.
"""
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
import numpy as np

DB_PATH = Path("data/sessions/sessions.db")


def _conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    conn = _conn(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            started_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active  TIMESTAMP,
            total_turns  INTEGER DEFAULT 0,
            dominant_topics TEXT,
            summary      TEXT
        );
        CREATE TABLE IF NOT EXISTS turns (
            turn_id             TEXT PRIMARY KEY,
            session_id          TEXT REFERENCES sessions(session_id),
            turn_index          INTEGER,
            timestamp           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_query          TEXT,
            assistant_response  TEXT,
            query_metadata_json TEXT,
            retrieved_chunk_ids TEXT,
            wikidata_context    TEXT
        );
        CREATE TABLE IF NOT EXISTS session_summaries (
            summary_id   TEXT PRIMARY KEY,
            session_id   TEXT,
            summary_text TEXT,
            embedding    BLOB,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
        CREATE INDEX IF NOT EXISTS idx_summaries_session ON session_summaries(session_id);
    """)
    conn.commit()
    conn.close()


def start_session(db_path: Path = DB_PATH) -> str:
    init_db(db_path)
    session_id = str(uuid.uuid4())
    conn = _conn(db_path)
    conn.execute(
        "INSERT INTO sessions (session_id, started_at, last_active, total_turns) VALUES (?,?,?,0)",
        (session_id, datetime.utcnow().isoformat(), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    return session_id


def save_turn(
    session_id: str,
    user_query: str,
    response: str,
    metadata: dict,
    chunk_ids: list[str],
    wikidata_ctx: str,
    db_path: Path = DB_PATH,
) -> str:
    conn = _conn(db_path)
    turn_row = conn.execute(
        "SELECT COUNT(*) FROM turns WHERE session_id=?", (session_id,)
    ).fetchone()[0]
    turn_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO turns VALUES (?,?,?,?,?,?,?,?,?)",
        (
            turn_id, session_id, turn_row,
            datetime.utcnow().isoformat(),
            user_query, response,
            json.dumps(metadata),
            json.dumps(chunk_ids),
            wikidata_ctx,
        ),
    )
    conn.execute(
        "UPDATE sessions SET last_active=?, total_turns=total_turns+1 WHERE session_id=?",
        (datetime.utcnow().isoformat(), session_id),
    )
    conn.commit()
    conn.close()
    return turn_id


def get_session_turns(session_id: str, db_path: Path = DB_PATH) -> list[dict]:
    conn = _conn(db_path)
    rows = conn.execute(
        "SELECT * FROM turns WHERE session_id=? ORDER BY turn_index ASC",
        (session_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_context(session_id: str, last_n: int = 5, db_path: Path = DB_PATH) -> str:
    turns = get_session_turns(session_id, db_path)[-last_n:]
    if not turns:
        return ""
    parts = ["CONVERSATION HISTORY"]
    for t in turns:
        parts.append(f"User: {t['user_query']}")
        parts.append(f"Scientist: {t['assistant_response']}")
        parts.append("---")
    return "\n".join(parts)


def end_session(
    session_id: str,
    summary_text: str,
    embedding: bytes,
    db_path: Path = DB_PATH,
) -> None:
    conn = _conn(db_path)
    conn.execute(
        "UPDATE sessions SET summary=? WHERE session_id=?",
        (summary_text, session_id),
    )
    conn.execute(
        "INSERT INTO session_summaries VALUES (?,?,?,?,?)",
        (str(uuid.uuid4()), session_id, summary_text, embedding,
         datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def search_past_sessions(
    query_embedding: bytes,
    top_k: int = 3,
    db_path: Path = DB_PATH,
) -> list[dict]:
    conn = _conn(db_path)
    rows = conn.execute(
        "SELECT session_id, summary_text, embedding, created_at FROM session_summaries"
    ).fetchall()
    conn.close()
    if not rows:
        return []

    q_vec = np.frombuffer(query_embedding, dtype=np.float32)
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)

    scored = []
    for r in rows:
        try:
            e_vec = np.frombuffer(r["embedding"], dtype=np.float32)
            e_norm = e_vec / (np.linalg.norm(e_vec) + 1e-9)
            sim = float(np.dot(q_norm, e_norm))
            scored.append({"session_id": r["session_id"], "summary": r["summary_text"],
                           "created_at": r["created_at"], "similarity": sim})
        except Exception:
            continue

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    init_db()
    sid = start_session()
    print(f"Started session: {sid}")
    t1 = save_turn(sid, "What is QED?", "QED is the theory of...",
                   {"question_type": "factual"}, ["c1", "c2"], "facts block")
    t2 = save_turn(sid, "How does it relate to Feynman diagrams?", "Feynman diagrams are...",
                   {"question_type": "interpretive"}, ["c3"], "facts block 2")
    ctx = get_recent_context(sid, last_n=2)
    print(f"\nRecent context:\n{ctx}")
    dummy_emb = np.zeros(768, dtype=np.float32).tobytes()
    end_session(sid, "Session about QED and Feynman diagrams.", dummy_emb)
    past = search_past_sessions(dummy_emb, top_k=1)
    print(f"\nPast sessions found: {len(past)}")
    print("✅ Session store smoke test passed")

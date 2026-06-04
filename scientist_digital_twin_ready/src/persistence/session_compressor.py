"""
Phase 5.3 - Session Compressor & Cross-Session Retriever.
LexRank compression + mpnet embeddings. No LLM.
"""
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# sumy depends on NLTK's punkt tokenizer — download once silently
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)  # required by NLTK >= 3.9

from src.persistence.session_store import (
    get_session_turns, end_session, search_past_sessions
)


class SessionCompressor:
    def __init__(self):
        self.summarizer = LexRankSummarizer()
        self.mpnet = SentenceTransformer("all-mpnet-base-v2")

    def compress_session(self, session_id: str) -> tuple[str, bytes]:
        turns = get_session_turns(session_id)
        if not turns:
            return "", np.zeros(768, dtype=np.float32).tobytes()

        full_text = "\n".join(
            f"User: {t['user_query']}\nScientist: {t['assistant_response']}"
            for t in turns
        )
        parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
        sentences = self.summarizer(parser.document, sentences_count=5)
        summary_text = " ".join(str(s) for s in sentences)

        embedding = (
            self.mpnet.encode(summary_text, normalize_embeddings=True)
            .astype(np.float32)
            .tobytes()
        )
        return summary_text, embedding

    def finalize_session(self, session_id: str) -> None:
        summary, embedding = self.compress_session(session_id)
        end_session(session_id, summary, embedding)
        print(f"✅ Session {session_id[:8]}... compressed: {len(summary)} chars")


class CrossSessionRetriever:
    def get_past_context(
        self,
        current_query: str,
        mpnet_model: SentenceTransformer,
        top_k: int = 2,
    ) -> str:
        query_emb = (
            mpnet_model.encode(current_query, normalize_embeddings=True)
            .astype(np.float32)
            .tobytes()
        )
        past = search_past_sessions(query_emb, top_k=top_k)
        if not past:
            return ""
        parts = ["RELEVANT PAST CONTEXT"]
        for p in past:
            date = p.get("created_at", "")[:10]
            parts.append(f"Session from {date}: {p['summary']}")
            parts.append("---")
        return "\n".join(parts)

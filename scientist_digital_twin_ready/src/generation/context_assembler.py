"""
Phase 6.1 - Context Window Assembler.
Builds system prompt + user message for each API call. No LLM.
"""
import json
import yaml
from pathlib import Path

PERSONA_BLOCK_PATH = Path("data/processed/persona_prompt_block.txt")
PROFILE_PATH = Path("data/processed/scientist_profile.json")
CONFIG_PATH = Path("config/scientist.yaml")

MAX_CONTEXT_CHARS = 12000


class ContextAssembler:
    def __init__(self):
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        self.scientist_name = config["scientist"]["name"]

        self.persona_block = (
            PERSONA_BLOCK_PATH.read_text(encoding="utf-8")
            if PERSONA_BLOCK_PATH.exists()
            else f"NAME: {self.scientist_name}"
        )
        self.profile = {}
        if PROFILE_PATH.exists():
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                self.profile = json.load(f)

    def _build_system_prompt(self) -> str:
        name = self.scientist_name
        return f"""You are {name}. Respond as {name} would — in first person, drawing on your documented knowledge, reasoning style, and characteristic vocabulary. Do not fabricate positions not evidenced in the provided context. When uncertain, hedge as {name} actually hedged. Keep responses conversational but precise — 2 to 4 paragraphs.

{self.persona_block}

INSTRUCTIONS: Answer the user's question grounded in the RETRIEVED CONTEXT provided. Prioritize direct evidence from your writings and lectures. If the question falls outside the provided context, say so as {name} would — with intellectual honesty rather than fabrication."""

    def _format_chunks(self, chunks: list[dict], max_chars_each: int = 400) -> str:
        if not chunks:
            return ""
        lines = ["RETRIEVED FROM YOUR WRITINGS"]
        for i, c in enumerate(chunks, 1):
            text = c["text"][:max_chars_each]
            source = c.get("source", "")
            year = c.get("year") or ""
            tag = f"{source} {year}".strip()
            lines.append(f"[{i}] ({tag})\n{text}")
        return "\n\n".join(lines)

    def _count_chars(self, *strings: str) -> int:
        return sum(len(s) for s in strings)

    def assemble(
        self,
        user_query: str,
        retrieval_result: dict,
        query_metadata: dict,
        session_history: str = "",
        past_session_context: str = "",
    ) -> tuple[str, str]:
        system_prompt = self._build_system_prompt()

        wikidata_ctx = retrieval_result.get("wikidata_context", "")
        chunks = retrieval_result.get("chunks", [])  # captured once for reuse in truncation
        chunks_text = self._format_chunks(chunks[:6])

        meta_lines = ["QUERY CONTEXT"]
        if query_metadata.get("scientific_concepts"):
            meta_lines.append(f"- Concepts: {', '.join(query_metadata['scientific_concepts'])}")
        meta_lines.append(f"- Type: {query_metadata.get('question_type', 'factual')}")
        meta_lines.append(f"- Tone: {query_metadata.get('emotional_tone', 'neutral')}")
        query_meta_str = "\n".join(meta_lines)

        def build_message(wk: str, ct: str, qm: str, psc: str, sh: str, uq: str) -> str:
            """Assemble user message from non-empty parts in priority order."""
            parts = [p for p in [wk, ct, qm, psc, sh] if p.strip()]
            parts.append(f"USER: {uq}")
            return "\n\n".join(parts)

        total_budget = MAX_CONTEXT_CHARS - len(system_prompt)

        # Step 1: Try with everything
        user_message = build_message(
            wikidata_ctx, chunks_text, query_meta_str,
            past_session_context, session_history, user_query,
        )
        if len(user_message) <= total_budget:
            return system_prompt, user_message

        # Step 2: Drop past_session_context (lowest priority)
        user_message = build_message(
            wikidata_ctx, chunks_text, query_meta_str,
            "", session_history, user_query,
        )
        if len(user_message) <= total_budget:
            return system_prompt, user_message

        # Step 3: Trim session_history to last 2 turns only
        turns = session_history.split("---")
        trimmed_history = "---".join(turns[-2:]) if len(turns) > 2 else session_history
        user_message = build_message(
            wikidata_ctx, chunks_text, query_meta_str,
            "", trimmed_history, user_query,
        )
        if len(user_message) <= total_budget:
            return system_prompt, user_message

        # Step 4: Drop session_history entirely, reduce to top-3 chunks at 300 chars each
        chunks_text = self._format_chunks(chunks[:3], max_chars_each=300)
        user_message = build_message(
            wikidata_ctx, chunks_text, query_meta_str,
            "", "", user_query,
        )
        # At this point we always return — never truncate wikidata or user query
        return system_prompt, user_message

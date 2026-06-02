import json
import os
from typing import List, Dict

MEMORY_FILE = "data/memory.json"

class MemoryManager:
    def __init__(self):
        self.short_term: List[Dict[str, str]] = []
        self.long_term: List[str] = self._load_long_term_memory()

    def _load_long_term_memory(self) -> List[str]:
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as f:
                    data = json.load(f)
                    return data.get("facts", [])
            except Exception:
                return []
        return []

    def _save_long_term_memory(self):
        with open(MEMORY_FILE, "w") as f:
            json.dump({"facts": self.long_term}, f, indent=4)

    def add_to_short_term(self, role: str, content: str):
        self.short_term.append({"role": role, "content": content})
        # Keep only the last 10 turns to prevent context window overflow
        if len(self.short_term) > 20:
            self.short_term = self.short_term[-20:]

    def get_short_term_context(self) -> str:
        if not self.short_term:
            return "No previous conversation."
        
        context = []
        for msg in self.short_term:
            role = "Feynman" if msg["role"] == "model" else "User"
            context.append(f"{role}: {msg['content']}")
        return "\n".join(context)

    def add_long_term_fact(self, fact: str):
        """Adds a fact about the user to long-term memory if it's new."""
        if fact and fact not in self.long_term:
            self.long_term.append(fact)
            self._save_long_term_memory()

    def get_long_term_context(self) -> str:
        if not self.long_term:
            return "No long-term memories stored yet."
        return "\n".join([f"- {fact}" for fact in self.long_term])

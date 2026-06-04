"""
Phase 5.1 - GLiNER Query Metadata Extractor.
Extracts structured metadata from user queries. No LLM.
"""
from gliner import GLiNER
from pathlib import Path

MODEL_NAME = "knowledgator/gliner-multitask-large-v0.5"

LABELS = [
    "scientific concept or theory",
    "person name",
    "time period or era",
    "question type: factual",
    "question type: interpretive",
    "question type: biographical",
    "question type: philosophical",
    "emotional tone: curious",
    "emotional tone: skeptical",
    "emotional tone: adversarial",
    "emotional tone: enthusiastic",
    "knowledge domain: physics",
    "knowledge domain: philosophy",
    "knowledge domain: biography",
    "knowledge domain: pedagogy",
    "specific work or paper",
    "comparison request",
]


class QueryMetadataExtractor:
    def __init__(self):
        self.model = GLiNER.from_pretrained(MODEL_NAME)

    def extract(self, user_query: str) -> dict:
        entities = self.model.predict_entities(user_query, LABELS, threshold=0.35)
        metadata = {
            "raw_query": user_query,
            "scientific_concepts": [],
            "persons_mentioned": [],
            "time_periods": [],
            "question_type": "factual",
            "emotional_tone": "neutral",
            "knowledge_domain": "general",
            "works_referenced": [],
            "is_comparison": False,
            "word_count": len(user_query.split()),
        }
        for ent in entities:
            label = ent["label"]
            text = ent["text"].strip()
            if "scientific concept" in label:
                metadata["scientific_concepts"].append(text)
            elif "person name" in label:
                metadata["persons_mentioned"].append(text)
            elif "time period" in label:
                metadata["time_periods"].append(text)
            elif "question type" in label:
                metadata["question_type"] = label.split(": ")[-1] if ": " in label else "factual"
            elif "emotional tone" in label:
                metadata["emotional_tone"] = label.split(": ")[-1] if ": " in label else "neutral"
            elif "knowledge domain" in label:
                metadata["knowledge_domain"] = label.split(": ")[-1] if ": " in label else "general"
            elif "specific work" in label:
                metadata["works_referenced"].append(text)
            elif "comparison" in label:
                metadata["is_comparison"] = True

        for key in ("scientific_concepts", "persons_mentioned", "time_periods", "works_referenced"):
            metadata[key] = list(dict.fromkeys(metadata[key]))
        return metadata

    def to_context_string(self, metadata: dict) -> str:
        lines = ["QUERY METADATA"]
        if metadata["scientific_concepts"]:
            lines.append(f"- Concepts: {', '.join(metadata['scientific_concepts'])}")
        if metadata["knowledge_domain"] != "general":
            lines.append(f"- Domain: {metadata['knowledge_domain']}")
        lines.append(f"- Type: {metadata['question_type']}")
        lines.append(f"- Tone: {metadata['emotional_tone']}")
        if metadata["works_referenced"]:
            lines.append(f"- Works referenced: {', '.join(metadata['works_referenced'])}")
        if metadata["is_comparison"]:
            lines.append("- Comparison request: yes")
        return "\n".join(lines)


if __name__ == "__main__":
    extractor = QueryMetadataExtractor()
    test_queries = [
        "Can you explain the path integral in simple terms?",
        "What did you think about Bohr's Copenhagen interpretation?",
        "How does QED compare to classical electrodynamics?",
        "Tell me about your time at Caltech and your students.",
        "Why do you reject hidden variable theories?",
    ]
    for q in test_queries:
        meta = extractor.extract(q)
        print(f"\nQuery: {q}")
        print(extractor.to_context_string(meta))

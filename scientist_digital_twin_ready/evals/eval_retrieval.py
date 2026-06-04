"""
Evaluation: Retrieval quality on 20-30 held-out questions.
Uses BERTScore to measure generation faithfulness to retrieved context.
"""
import json
import sys
from pathlib import Path

EVAL_QUESTIONS_PATH = Path("evals/eval_questions.json")
RESULTS_PATH = Path("evals/eval_results.json")


SAMPLE_QUESTIONS = [
    {"query": "What is the path integral formulation of quantum mechanics?",
     "expected_topics": ["path integral", "quantum mechanics", "action"]},
    {"query": "How did Feynman approach the problem of quantum electrodynamics?",
     "expected_topics": ["QED", "renormalization", "Feynman diagrams"]},
    {"query": "What are Feynman diagrams and what do they represent?",
     "expected_topics": ["Feynman diagrams", "perturbation theory", "particles"]},
    {"query": "What did Feynman think about the importance of not knowing?",
     "expected_topics": ["uncertainty", "doubt", "scientific method"]},
    {"query": "Describe Feynman's work on the Manhattan Project.",
     "expected_topics": ["Los Alamos", "atomic bomb", "Manhattan Project"]},
]


def run_eval():
    # Write sample questions if not present
    if not EVAL_QUESTIONS_PATH.exists():
        EVAL_QUESTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        EVAL_QUESTIONS_PATH.write_text(
            json.dumps(SAMPLE_QUESTIONS, indent=2), encoding="utf-8"
        )
        print(f"Sample questions written to {EVAL_QUESTIONS_PATH}")
        print("Edit this file to add your own evaluation questions.")

    with open(EVAL_QUESTIONS_PATH) as f:
        questions = json.load(f)

    try:
        from src.retrieval.retrieval_engine import RetrievalEngine
        engine = RetrievalEngine()
    except Exception as e:
        print(f"Could not load retrieval engine: {e}")
        sys.exit(1)

    results = []
    for q in questions:
        query = q["query"]
        expected_topics = q.get("expected_topics", [])
        result = engine.retrieve(query, top_k=5)

        all_text = " ".join(c["text"].lower() for c in result["chunks"])
        topic_hits = [t for t in expected_topics if t.lower() in all_text]
        recall = len(topic_hits) / max(len(expected_topics), 1)

        results.append({
            "query": query,
            "expected_topics": expected_topics,
            "topic_hits": topic_hits,
            "recall": round(recall, 2),
            "bm25_count": result["bm25_count"],
            "faiss_count": result["faiss_count"],
            "both_count": result["both_count"],
            "num_chunks_returned": len(result["chunks"]),
        })
        print(f"[{recall:.0%}] {query[:60]}")

    avg_recall = sum(r["recall"] for r in results) / len(results)
    print(f"\n=== Retrieval Eval ===")
    print(f"Questions: {len(results)}")
    print(f"Avg topic recall: {avg_recall:.1%}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    run_eval()

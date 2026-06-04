"""
Evaluation: Generation faithfulness using BERTScore.
Measures how well generated responses are grounded in retrieved context.
"""
import json
import os
from pathlib import Path

EVAL_QA_PATH = Path("evals/eval_qa_pairs.json")
RESULTS_PATH = Path("evals/generation_eval_results.json")

SAMPLE_QA_PAIRS = [
    {
        "query": "What is the path integral?",
        "reference": "The path integral formulation integrates over all possible paths a particle could take, weighting each by its phase factor determined by the action."
    },
    {
        "query": "How does QED work?",
        "reference": "QED describes how light and matter interact through the exchange of photons, using Feynman diagrams to represent terms in perturbation expansions."
    },
]


def run_eval():
    if not EVAL_QA_PATH.exists():
        EVAL_QA_PATH.parent.mkdir(parents=True, exist_ok=True)
        EVAL_QA_PATH.write_text(json.dumps(SAMPLE_QA_PAIRS, indent=2), encoding="utf-8")
        print(f"Sample QA pairs written to {EVAL_QA_PATH}")
        print("Add real (query, reference_answer) pairs to evaluate generation.")
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY to run generation eval.")
        return

    try:
        from bert_score import score as bert_score
    except ImportError:
        print("Install bert-score: pip install bert-score")
        return

    with open(EVAL_QA_PATH) as f:
        pairs = json.load(f)

    from src.generation.generate import GenerationHandler
    from src.persistence.session_store import start_session

    handler = GenerationHandler()
    session_id = start_session()

    candidates = []
    references = []
    for pair in pairs:
        response = handler.generate(pair["query"], session_id, text_only=True)
        candidates.append(response)
        references.append(pair["reference"])
        print(f"Q: {pair['query'][:60]}")
        print(f"A: {response[:100]}...\n")

    P, R, F1 = bert_score(candidates, references, lang="en", verbose=False)
    results = {
        "num_questions": len(pairs),
        "avg_bertscore_f1": float(F1.mean()),
        "avg_bertscore_precision": float(P.mean()),
        "avg_bertscore_recall": float(R.mean()),
        "per_question": [
            {"query": p["query"], "f1": float(f1), "precision": float(p_), "recall": float(r_)}
            for p, f1, p_, r_ in zip(pairs, F1, P, R)
        ],
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n=== Generation Eval ===")
    print(f"Avg BERTScore F1: {results['avg_bertscore_f1']:.3f}")
    print(f"Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    run_eval()

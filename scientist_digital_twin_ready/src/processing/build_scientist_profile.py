"""
Phase 3.1 - GLiNER Profile Extractor.
Extracts scientist's intellectual profile from corpus. No LLM.
"""
import json
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from gliner import GLiNER

CHUNKS_PATH = Path("data/processed/all_chunks.json")
OUTPUT_PATH = Path("data/processed/scientist_profile_gliner.json")
CHECKPOINT_PATH = Path("data/processed/gliner_checkpoint.json")

MODEL_NAME = "knowledgator/gliner-multitask-large-v0.5"
THRESHOLD = 0.4
BATCH_SIZE = 20

LABELS = [
    "scientific concept",
    "physics theory or framework",
    "mathematical method or formalism",
    "experimental technique",
    "scientific claim or position",
    "epistemic hedge phrase",
    "rejected theory or idea",
    "characteristic analogy",
    "signature vocabulary",
    "collaborator name",
    "institution name",
    "time period reference",
]


def process_chunks(chunks: list[dict], model: GLiNER) -> dict:
    scientific_concepts = Counter()
    theories_frameworks = []
    methods_formalisms = []
    explicit_positions = []
    hedge_phrases = Counter()
    rejected_ideas = []
    characteristic_analogies = []
    signature_vocabulary = Counter()
    collaborators = []

    checkpoint = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
        print(f"Resuming from checkpoint: {len(checkpoint.get('processed', []))} chunks done")

    processed_ids = set(checkpoint.get("processed", []))

    for i, chunk in enumerate(tqdm(chunks, desc="GLiNER extraction")):
        cid = chunk["chunk_id"]
        if cid in processed_ids:
            continue
        try:
            entities = model.predict_entities(
                chunk["text"][:2000],  # cap per-chunk input
                LABELS,
                threshold=THRESHOLD,
            )
            for ent in entities:
                label = ent["label"]
                text = ent["text"].strip()
                if len(text) < 2:
                    continue
                if "scientific concept" in label:
                    scientific_concepts[text] += 1
                elif "physics theory" in label or "framework" in label:
                    theories_frameworks.append(text)
                elif "mathematical method" in label or "formalism" in label:
                    methods_formalisms.append(text)
                elif "experimental technique" in label:
                    methods_formalisms.append(text)
                elif "claim or position" in label:
                    explicit_positions.append({"text": text, "source_chunk_id": cid})
                elif "epistemic hedge" in label:
                    hedge_phrases[text] += 1
                elif "rejected theory" in label:
                    rejected_ideas.append(text)
                elif "characteristic analogy" in label:
                    characteristic_analogies.append(text)
                elif "signature vocabulary" in label:
                    signature_vocabulary[text] += 1
                elif "collaborator" in label:
                    collaborators.append(text)
            processed_ids.add(cid)
            # Save checkpoint every 100 chunks
            if len(processed_ids) % 100 == 0:
                CHECKPOINT_PATH.write_text(
                    json.dumps({"processed": list(processed_ids)}), encoding="utf-8"
                )
        except Exception as e:
            print(f"  ⚠️  Chunk {cid[:8]} failed: {e}")
            continue

    total_chunks = len(chunks)
    total_hedge_instances = sum(hedge_phrases.values())
    hedge_ratio = total_hedge_instances / max(total_chunks, 1)

    return {
        "scientific_concepts": dict(scientific_concepts.most_common(50)),
        "theories_frameworks": list(set(theories_frameworks)),
        "methods_formalisms": list(set(methods_formalisms)),
        "explicit_positions": explicit_positions,
        "hedge_phrases": dict(hedge_phrases.most_common(20)),
        "rejected_ideas": list(set(rejected_ideas)),
        "characteristic_analogies": list(set(characteristic_analogies)),
        "signature_vocabulary": dict(signature_vocabulary.most_common(30)),
        "collaborators": list(set(collaborators)),
        "hedge_ratio": round(hedge_ratio, 4),
        "avg_concepts_per_chunk": round(sum(scientific_concepts.values()) / max(total_chunks, 1), 2),
    }


def main():
    print("Loading chunks...")
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading GLiNER model: {MODEL_NAME}")
    model = GLiNER.from_pretrained(MODEL_NAME)

    profile = process_chunks(chunks, model)
    OUTPUT_PATH.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✅ Profile saved → {OUTPUT_PATH}")
    print(f"  Top 10 concepts: {list(profile['scientific_concepts'].keys())[:10]}")
    print(f"  Hedge ratio: {profile['hedge_ratio']} ({'high' if profile['hedge_ratio'] > 0.15 else 'moderate'})")
    print(f"  Explicit positions: {len(profile['explicit_positions'])}")
    print(f"  Rejected ideas: {len(profile['rejected_ideas'])}")

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()


if __name__ == "__main__":
    main()

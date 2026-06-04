"""
Phase 3.2 - Scientist Profile Merger.
Combines Wikidata + GLiNER + papers manifest into final persona artifact.
No LLM.
"""
import json
from pathlib import Path

WIKIDATA_PATH = Path("data/processed/scientist_wikidata.json")
GLINER_PATH = Path("data/processed/scientist_profile_gliner.json")
MANIFEST_PATH = Path("data/processed/papers/manifest.json")
OUTPUT_PATH = Path("data/processed/scientist_profile.json")
PERSONA_BLOCK_PATH = Path("data/processed/persona_prompt_block.txt")


def word_overlap(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def deduplicate(items: list[str], threshold: float = 0.8) -> list[str]:
    unique = []
    for item in items:
        if not any(word_overlap(item, u) > threshold for u in unique):
            unique.append(item)
    return unique


def main():
    with open(WIKIDATA_PATH) as f:
        wikidata = json.load(f)
    with open(GLINER_PATH) as f:
        gliner = json.load(f)
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # Biographical spine from Wikidata
    bio_fields = ["name", "qid", "description", "date_of_birth", "date_of_death",
                  "country_of_citizenship", "place_of_birth", "gender"]
    bio = {k: wikidata.get(k) for k in bio_fields}

    def extract_list_values(key: str) -> list[str]:
        items = wikidata.get(key, [])
        return [v["value"] if isinstance(v, dict) else str(v) for v in items]

    # Determine epistemic style
    hedge_ratio = gliner.get("hedge_ratio", 0)
    if hedge_ratio > 0.15:
        style_descriptor = "hedges frequently, shows epistemic humility"
    elif hedge_ratio > 0.05:
        style_descriptor = "moderate confidence, occasional hedging"
    else:
        style_descriptor = "high confidence assertions"

    # Birth/death year
    dob = bio.get("date_of_birth", "") or ""
    dod = bio.get("date_of_death", "") or ""
    birth_year = dob[:4] if dob else "?"
    death_year = dod[:4] if dod else "present"

    # Most cited works from manifest
    papers_sorted = sorted(
        manifest.get("papers", []),
        key=lambda p: p.get("citation_count", 0),
        reverse=True,
    )
    most_cited = [{"title": p["title"], "year": p.get("year"), "citations": p.get("citation_count")}
                  for p in papers_sorted[:5]]

    # Publication span
    years = [p["year"] for p in manifest.get("papers", []) if p.get("year")]
    pub_span = f"{min(years)}-{max(years)}" if years else "unknown"

    profile = {
        "biographical_spine": bio,
        "era": f"{birth_year}–{death_year}",
        "institutions": extract_list_values("employer"),
        "doctoral_advisor": extract_list_values("doctoral_advisor"),
        "doctoral_students": extract_list_values("doctoral_students"),
        "awards": extract_list_values("awards_received"),
        "notable_works": extract_list_values("notable_works"),
        "influenced_by": extract_list_values("influenced_by"),
        "influenced": extract_list_values("influenced"),
        "primary_fields": list(gliner.get("scientific_concepts", {}).keys())[:10],
        "theories": gliner.get("theories_frameworks", []),
        "methods": gliner.get("methods_formalisms", []),
        "known_positions": [p["text"] for p in gliner.get("explicit_positions", [])],
        "rejected_ideas": deduplicate(gliner.get("rejected_ideas", [])),
        "analogies_used": gliner.get("characteristic_analogies", []),
        "signature_vocabulary": list(gliner.get("signature_vocabulary", {}).keys())[:20],
        "epistemic_style": {
            "style_descriptor": style_descriptor,
            "hedge_ratio": hedge_ratio,
            "hedging_phrases": list(gliner.get("hedge_phrases", {}).keys())[:10],
        },
        "publication_record": {
            "total_papers": manifest.get("total_papers", 0),
            "publication_span": pub_span,
            "most_cited_works": most_cited,
        },
    }

    OUTPUT_PATH.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Profile saved → {OUTPUT_PATH}")

    # Generate persona prompt block
    name = bio.get("name", "Scientist")
    lines = [
        "SCIENTIST PROFILE",
        f"NAME: {name}",
        f"FIELDS: {', '.join(profile['primary_fields'])}",
        f"ERA: {profile['era']}",
        f"INSTITUTIONS: {', '.join(profile['institutions'][:4])}",
        f"NOTABLE WORKS: {', '.join(profile['notable_works'][:5])}",
        f"KEY COLLABORATORS: {', '.join((profile['doctoral_advisor'] + profile['doctoral_students'] + profile['influenced_by'])[:8])}",
        f"CORE THEORIES & METHODS: {', '.join((profile['theories'] + profile['methods'])[:10])}",
        "",
        "KNOWN POSITIONS:",
        *[f"- {p}" for p in profile["known_positions"][:10]],
        "",
        f"IDEAS THEY REJECTED: {', '.join(profile['rejected_ideas'][:5])}",
        f"CHARACTERISTIC ANALOGIES: {', '.join(profile['analogies_used'][:5])}",
        f"SIGNATURE VOCABULARY: {', '.join(profile['signature_vocabulary'])}",
        f"EPISTEMIC STYLE: {profile['epistemic_style']['style_descriptor']}",
        f"HEDGING PHRASES THEY USE: {', '.join(profile['epistemic_style']['hedging_phrases'])}",
    ]
    PERSONA_BLOCK_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Persona block saved → {PERSONA_BLOCK_PATH}")
    print("\n" + "\n".join(lines[:15]))


if __name__ == "__main__":
    main()

"""
Phase 1.3 - Wikipedia Article Fetcher.
Extracts full article by section. No LLM.
"""
import re
import json
import yaml
from pathlib import Path
import wikipediaapi

CONFIG_PATH = Path("config/scientist.yaml")
OUT_FULL = Path("data/raw/transcripts/wikipedia_full.txt")
OUT_SECTIONS = Path("data/processed/wikipedia_sections.json")
OUT_TALK = Path("data/raw/transcripts/wikipedia_talk.txt")


def strip_citations(text: str) -> str:
    return re.sub(r"\[\d+\]", "", text).strip()


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    title = config["scientist"]["wikipedia_title"]

    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="ScientistTwin/1.0 (research project)",
    )

    page = wiki.page(title)
    if not page.exists():
        raise ValueError(f"Wikipedia page not found: {title}")

    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    OUT_SECTIONS.parent.mkdir(parents=True, exist_ok=True)

    sections_data = []
    full_lines = []

    def process_section(section, level=0):
        text = strip_citations(section.text)
        wc = len(text.split())
        if text:
            sections_data.append({"title": section.title, "text": text, "word_count": wc, "level": level})
            full_lines.append(f"\n\nSECTION: {section.title}\n{text}")
            print(f"  {'  '*level}{section.title}: {wc} words")
        for sub in section.sections:
            process_section(sub, level + 1)

    # Summary (intro)
    intro = strip_citations(page.summary)
    sections_data.insert(0, {"title": "Introduction", "text": intro, "word_count": len(intro.split()), "level": 0})
    full_lines.insert(0, f"SECTION: Introduction\n{intro}")
    print(f"  Introduction: {len(intro.split())} words")

    for section in page.sections:
        process_section(section)

    OUT_FULL.write_text("\n".join(full_lines), encoding="utf-8")
    OUT_SECTIONS.write_text(json.dumps(sections_data, indent=2, ensure_ascii=False), encoding="utf-8")

    total_words = sum(s["word_count"] for s in sections_data)
    print(f"\n✅ Wikipedia: {len(sections_data)} sections, {total_words} total words → {OUT_FULL}")

    # Talk page (best effort)
    talk_page = wiki.page(f"Talk:{title}")
    if talk_page.exists():
        OUT_TALK.write_text(strip_citations(talk_page.text), encoding="utf-8")
        print(f"✅ Talk page saved → {OUT_TALK}")
    else:
        print("ℹ️  No talk page available.")


if __name__ == "__main__":
    main()

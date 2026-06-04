"""
Phase 1.4 - YouTube Transcript Fetcher.
Downloads and segments transcripts. No LLM.
"""
import json
import re
import yaml
import requests
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

CONFIG_PATH = Path("config/scientist.yaml")
OUT_DIR = Path("data/raw/transcripts")
MANIFEST_PATH = Path("data/processed/transcripts_manifest.json")


def get_video_title(video_id: str) -> str:
    try:
        r = requests.get(
            f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json",
            timeout=8,
        )
        return r.json().get("title", video_id)
    except Exception:
        return video_id


def merge_into_paragraphs(entries: list[dict], target_words: int = 200) -> list[dict]:
    paragraphs = []
    current_text = []
    current_start = None
    word_count = 0
    for entry in entries:
        text = re.sub(r"\[.*?\]", "", entry["text"]).strip()  # strip [Music], [Applause]
        if not text:
            continue
        if current_start is None:
            current_start = entry["start"]
        current_text.append(text)
        word_count += len(text.split())
        if word_count >= target_words:
            paragraphs.append({"text": " ".join(current_text), "start": current_start})
            current_text = []
            current_start = None
            word_count = 0
    if current_text:
        paragraphs.append({"text": " ".join(current_text), "start": current_start or 0})
    return paragraphs


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    video_ids = config["scientist"].get("youtube_video_ids", [])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest = []
    for vid in video_ids:
        print(f"\nProcessing video: {vid}")
        title = get_video_title(vid)
        print(f"  Title: {title}")

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(vid)
            try:
                transcript = transcript_list.find_manually_created_transcript(["en"])
            except Exception:
                transcript = transcript_list.find_generated_transcript(["en"])
            entries = transcript.fetch()
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            print(f"  ⚠️  No transcript: {e}")
            continue
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            continue

        paragraphs = merge_into_paragraphs(entries)
        lines = []
        prev_timestamp_at = 0
        for para in paragraphs:
            start = para["start"]
            if start - prev_timestamp_at >= 60:
                lines.append(f"\n[{format_timestamp(start)}]")
                prev_timestamp_at = start
            lines.append(para["text"])

        out_file = OUT_DIR / f"{vid}.txt"
        out_file.write_text("\n".join(lines), encoding="utf-8")

        duration_seconds = entries[-1]["start"] + entries[-1].get("duration", 0) if entries else 0
        word_count = sum(len(p["text"].split()) for p in paragraphs)
        manifest.append({
            "video_id": vid,
            "title": title,
            "word_count": word_count,
            "duration_minutes": round(duration_seconds / 60, 1),
            "filepath": str(out_file),
        })
        print(f"  ✅ {word_count} words, {round(duration_seconds/60,1)} min → {out_file}")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Manifest saved → {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

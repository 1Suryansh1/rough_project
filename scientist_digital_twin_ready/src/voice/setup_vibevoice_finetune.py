"""
Phase 7.2A - VibeVoice Fine-Tune Setup.
Clones community fork and prepares dataset. No LLM.
"""
import json
import subprocess
import shutil
from pathlib import Path

VOICE_MANIFEST = Path("data/processed/voice_manifest.json")
FINETUNE_DIR = Path("data/voice/finetune_dataset")
VIBEVOICE_REPO = Path("data/voice/VibeVoice")
COMMUNITY_FORK_URL = "https://github.com/vibevoice-community/VibeVoice.git"


def clone_repo():
    if VIBEVOICE_REPO.exists():
        print(f"VibeVoice repo already exists at {VIBEVOICE_REPO}")
        return
    print("Cloning VibeVoice community fork...")
    subprocess.run(
        ["git", "clone", COMMUNITY_FORK_URL, str(VIBEVOICE_REPO)],
        check=True,
    )
    print(f"✅ Cloned to {VIBEVOICE_REPO}")


def prepare_dataset():
    if not VOICE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Voice manifest not found: {VOICE_MANIFEST}\n"
            "Run src/voice/prepare_voice_data.py first."
        )

    with open(VOICE_MANIFEST) as f:
        manifest = json.load(f)

    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    wavs_dir = FINETUNE_DIR / "wavs"
    wavs_dir.mkdir(exist_ok=True)

    metadata_lines = []
    for seg in manifest["segments"]:
        src_audio = Path(seg["audio_path"])
        src_text = Path(seg["text_path"])
        if not src_audio.exists() or not src_text.exists():
            continue
        dest_audio = wavs_dir / src_audio.name
        shutil.copy2(src_audio, dest_audio)
        text = src_text.read_text(encoding="utf-8").strip()
        # LJSpeech-style metadata: filename|transcript
        metadata_lines.append(f"{dest_audio.stem}|{text}")

    metadata_path = FINETUNE_DIR / "metadata.csv"
    metadata_path.write_text("\n".join(metadata_lines), encoding="utf-8")
    print(f"✅ Dataset prepared: {len(metadata_lines)} samples → {FINETUNE_DIR}")
    print(f"   metadata.csv: {metadata_path}")

    # Write dataset config
    dataset_config = {
        "dataset_path": str(FINETUNE_DIR),
        "metadata_file": str(metadata_path),
        "wavs_dir": str(wavs_dir),
        "total_samples": len(metadata_lines),
        "format": "ljspeech",
        "sample_rate": 16000,
    }
    (FINETUNE_DIR / "dataset_config.json").write_text(
        json.dumps(dataset_config, indent=2), encoding="utf-8"
    )
    return dataset_config


def main():
    clone_repo()
    config = prepare_dataset()
    print("\n--- Dataset Config ---")
    print(json.dumps(config, indent=2))
    print("\nNext step: Run src/voice/finetune_vibevoice.sh")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Full pipeline runner for Scientist Digital Twin.
Run each phase in sequence. Skip phases already completed.

Usage:
  python run_pipeline.py --phase all        # Run all phases
  python run_pipeline.py --phase ingestion  # Phase 1 only
  python run_pipeline.py --phase processing # Phase 2 only
  python run_pipeline.py --phase profile    # Phase 3 only
  python run_pipeline.py --phase voice      # Phase 7 only
  python run_pipeline.py --chat             # Start interactive chat
"""
import argparse
import sys
from pathlib import Path

def phase_done(marker: str) -> bool:
    return Path(marker).exists()

def run_ingestion():
    print("\n" + "="*50)
    print("PHASE 1: DATA INGESTION")
    print("="*50)
    from src.ingestion.fetch_wikidata import main as wikidata_main
    from src.ingestion.fetch_semantic_scholar import main as ss_main
    from src.ingestion.fetch_wikipedia import main as wiki_main
    from src.ingestion.fetch_youtube_transcripts import main as yt_main
    from src.ingestion.fetch_supplementary import main as supp_main
    wikidata_main()
    ss_main()
    wiki_main()
    yt_main()
    supp_main()

def run_processing():
    print("\n" + "="*50)
    print("PHASE 2: CORPUS PROCESSING")
    print("="*50)
    from src.processing.chunker import main as chunk_main
    from src.processing.build_bm25_index import build_index as bm25_main
    from src.processing.build_faiss_index import build_index as faiss_main
    from src.processing.chunk_store import populate_db
    chunk_main()
    bm25_main()
    faiss_main()
    populate_db()

def run_profile():
    print("\n" + "="*50)
    print("PHASE 3: SCIENTIST PROFILE")
    print("="*50)
    from src.processing.build_scientist_profile import main as gliner_main
    from src.processing.merge_scientist_profile import main as merge_main
    gliner_main()
    merge_main()

def run_voice():
    print("\n" + "="*50)
    print("PHASE 7: VOICE PIPELINE")
    print("="*50)
    from src.voice.prepare_voice_data import main as voice_main
    from src.voice.setup_vibevoice_finetune import main as setup_main
    voice_main()
    setup_main()
    print("\nTo launch fine-tuning: bash src/voice/finetune_vibevoice.sh")

def run_chat():
    from src.generation.generate import main as gen_main
    gen_main()

def main():
    parser = argparse.ArgumentParser(description="Scientist Digital Twin Pipeline Runner")
    parser.add_argument("--phase", choices=["all", "ingestion", "processing", "profile", "voice"],
                        help="Which phase to run")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat")
    args = parser.parse_args()

    if args.chat:
        run_chat()
        return

    if not args.phase:
        parser.print_help()
        return

    if args.phase in ("all", "ingestion"):
        run_ingestion()
    if args.phase in ("all", "processing"):
        run_processing()
    if args.phase in ("all", "profile"):
        run_profile()
    if args.phase in ("voice",):
        run_voice()
    if args.phase == "all":
        print("\n✅ Full pipeline complete. Run 'python run_pipeline.py --chat' to start chatting.")

if __name__ == "__main__":
    main()

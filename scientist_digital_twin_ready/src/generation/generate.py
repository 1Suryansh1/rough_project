"""
Phase 6.2 - Anthropic Generation Handler.
The ONLY place an LLM is used in the entire system.
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import argparse
import yaml
import google.generativeai as genai
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any

from src.retrieval.retrieval_engine import RetrievalEngine
from src.persistence.metadata_extractor import QueryMetadataExtractor
from src.persistence.session_store import start_session, save_turn, get_recent_context
from src.persistence.session_compressor import CrossSessionRetriever, SessionCompressor
from src.generation.context_assembler import ContextAssembler

CONFIG_PATH = Path("config/scientist.yaml")
MODEL = "gemini-2.5-flash"
MAX_TOKENS = 800


class GenerationHandler:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable not set. "
                "Export it before running: export GEMINI_API_KEY=your-gemini-key..."
            )
        genai.configure(api_key=api_key)

        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        self.scientist_name = config["scientist"]["name"]

        print("Loading retrieval engine...")
        self.retrieval_engine = RetrievalEngine()
        print("Loading metadata extractor...")
        self.metadata_extractor = QueryMetadataExtractor()
        print("Loading context assembler...")
        self.assembler = ContextAssembler()
        print("Loading mpnet for session embeddings...")
        self.mpnet = SentenceTransformer("all-mpnet-base-v2")
        self.cross_session = CrossSessionRetriever()
        print("✅ GenerationHandler ready.\n")

    def generate(self, user_query: str, session_id: str, text_only: bool = False) -> str:
        # Step 1: Extract query metadata
        query_metadata = self.metadata_extractor.extract(user_query)

        # Step 2: Retrieve relevant chunks + wikidata context
        retrieval_result = self.retrieval_engine.retrieve(user_query, top_k=8)

        # Step 3: Get session history
        session_history = get_recent_context(session_id, last_n=4)

        # Step 4: Get cross-session past context
        past_context = self.cross_session.get_past_context(
            user_query, self.mpnet, top_k=2
        )

        # Step 5: Assemble context window
        system_prompt, user_message = self.assembler.assemble(
            user_query=user_query,
            retrieval_result=retrieval_result,
            query_metadata=query_metadata,
            session_history=session_history,
            past_session_context=past_context,
        )

        # Step 6: API call — THE ONLY LLM CALL IN THE SYSTEM
        model = genai.GenerativeModel(MODEL, system_instruction=system_prompt)
        response = model.generate_content(
            user_message,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=MAX_TOKENS,
            )
        )
        assistant_text = response.text

        # Step 7: Persist turn
        save_turn(
            session_id=session_id,
            user_query=user_query,
            response=assistant_text,
            metadata=query_metadata,
            chunk_ids=[c["chunk_id"] for c in retrieval_result["chunks"]],
            wikidata_ctx=retrieval_result["wikidata_context"],
        )

        if text_only:
            return assistant_text

        # Step 8: Synthesize voice response and auto-play
        try:
            from src.voice.synthesize import synthesize_audio
            import sounddevice as sd
            import soundfile as sf
            
            out_wav = os.path.join("data", "processed", "voice_outputs", f"response_{session_id[-8:]}.wav")
            print("\n🎙️ Cloning voice (zero-shot F5-TTS)...")
            synthesize_audio(
                text=assistant_text,
                ref_audio_path="data/processed/voice_samples/feynman_reference.wav",
                ref_text_path="data/processed/voice_samples/feynman_reference.txt",
                out_path=out_wav
            )
            print(f"🔈 Voice generated: {out_wav}")
            
            print("🔊 Playing response...")
            data, fs = sf.read(out_wav)
            sd.play(data, fs)
            sd.wait()
            
        except Exception as e:
            print(f"⚠️ Voice synthesis or playback failed: {e}")

        return assistant_text

    def start_new_session(self) -> str:
        return start_session()


def main():
    parser = argparse.ArgumentParser(description="Scientist Digital Twin CLI")
    parser.add_argument("--session-id", type=str, default=None,
                        help="Resume an existing session by ID")
    parser.add_argument("--query", type=str, default=None,
                        help="Single query (non-interactive mode)")
    parser.add_argument("--text-only", action="store_true",
                        help="Skip voice synthesis and just output text")
    args = parser.parse_args()

    handler = GenerationHandler()

    if args.session_id:
        session_id = args.session_id
        print(f"Resuming session: {session_id}")
    else:
        session_id = handler.start_new_session()
        print(f"New session started: {session_id}")

    scientist_name = handler.scientist_name

    if args.query:
        response = handler.generate(args.query, session_id, text_only=args.text_only)
        print(f"\n{scientist_name}:\n{response}\n")
        return

    # Interactive loop
    print(f"\nTalking to {scientist_name}. Type 'quit' or 'exit' to end session.\n")
    compressor = SessionCompressor()
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                break
            response = handler.generate(user_input, session_id)
            print(f"\n{scientist_name}:\n{response}\n")
    finally:
        print("\nCompressing session...")
        compressor.finalize_session(session_id)
        print(f"Session {session_id} saved.")


if __name__ == "__main__":
    main()

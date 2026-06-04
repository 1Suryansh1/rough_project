import os
import argparse
import soundfile as sf
import torch
import torchaudio

# Completely bypass torchaudio.load to fix Windows torchcodec crash
def patched_load(filepath, *args, **kwargs):
    wav, sr = sf.read(filepath)
    if wav.ndim == 1:
        wav = wav.reshape(1, -1)
    else:
        wav = wav.T # channels first
    return torch.tensor(wav, dtype=torch.float32), sr

torchaudio.load = patched_load

try:
    from f5_tts.api import F5TTS
except ImportError:
    print("F5TTS not found. Please install f5-tts.")
    exit(1)

def synthesize_audio(text, ref_audio_path, ref_text_path, out_path):
    if not os.path.exists(ref_audio_path):
        raise FileNotFoundError(f"Reference audio missing: {ref_audio_path}")
    if not os.path.exists(ref_text_path):
        raise FileNotFoundError(f"Reference text missing: {ref_text_path}")
        
    print(f"Loading F5-TTS... (CUDA available: {torch.cuda.is_available()})")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize the zero-shot F5-TTS model
    f5tts = F5TTS(device=device)
    
    print(f"Reading reference text from: {ref_text_path}")
    with open(ref_text_path, "r", encoding="utf-8") as f:
        ref_text = f.read().strip()
    
    print(f"Generating zero-shot clone for text:\n{text}\n")
    wav, sr, spect = f5tts.infer(
        ref_file=ref_audio_path,
        ref_text=ref_text,
        gen_text=text
    )
    
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, wav, sr)
    print(f"Successfully saved synthesized audio to: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--out", type=str, default="data/processed/output.wav")
    parser.add_argument("--ref-audio", type=str, default="data/processed/voice_samples/feynman_reference.wav")
    parser.add_argument("--ref-text", type=str, default="data/processed/voice_samples/feynman_reference.txt")
    args = parser.parse_args()
    
    synthesize_audio(args.text, args.ref_audio, args.ref_text, args.out)

if __name__ == "__main__":
    main()

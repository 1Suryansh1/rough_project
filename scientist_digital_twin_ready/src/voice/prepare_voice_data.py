import os
import soundfile as sf
from datasets import load_dataset

def main():
    out_dir = os.path.join("data", "processed", "voice_samples")
    os.makedirs(out_dir, exist_ok=True)
    
    out_wav = os.path.join(out_dir, "feynman_reference.wav")
    out_txt = os.path.join(out_dir, "feynman_reference.txt")
    
    print("Hunting for 10-sec audio of Richard Feynman...")
    print("Downloading authentic clip from HuggingFace dataset 'T-T-S/FunToImagineWithRichardFeynmanAudioClips'...")
    
    try:
        # Load the first row of the dataset
        ds = load_dataset("T-T-S/FunToImagineWithRichardFeynmanAudioClips", split="train")
        sample = ds[0]
        
        # Extract audio array and sample rate
        audio_array = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        
        # Extract the exact transcript
        transcript = sample.get("text", sample.get("sentence", sample.get("transcript", "")))
        
        # Save the audio as wav
        sf.write(out_wav, audio_array, sr)
        print(f"✅ Successfully extracted and saved zero-shot reference audio: {out_wav}")
        
        # Save the exact text
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"✅ Saved zero-shot reference text: {out_txt}")
        print(f"Transcript: \"{transcript}\"")
        
    except Exception as e:
        print(f"Failed to extract audio: {e}")

if __name__ == "__main__":
    main()

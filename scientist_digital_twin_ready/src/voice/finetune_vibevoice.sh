#!/bin/bash
# Phase 7.2B - VibeVoice LoRA Fine-Tune Launch Script
# Requirements: >=16GB VRAM for 1.5B model
# Run from project root: bash src/voice/finetune_vibevoice.sh

set -e

REPO_DIR="data/voice/VibeVoice"
DATASET_DIR="data/voice/finetune_dataset"
OUTPUT_DIR="data/voice/lora_checkpoint"

echo "=== VibeVoice LoRA Fine-Tune ==="
echo "Repo:    $REPO_DIR"
echo "Dataset: $DATASET_DIR"
echo "Output:  $OUTPUT_DIR"
echo ""

if [ ! -d "$REPO_DIR" ]; then
  echo "ERROR: VibeVoice repo not found. Run setup_vibevoice_finetune.py first."
  exit 1
fi

if [ ! -f "$DATASET_DIR/metadata.csv" ]; then
  echo "ERROR: Dataset not prepared. Run setup_vibevoice_finetune.py first."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Install fine-tuning dependencies from community fork
cd "$REPO_DIR"
if [ -f "requirements_finetune.txt" ]; then
  pip install -q -r requirements_finetune.txt
fi
cd -

# Launch LoRA fine-tune
# The community fork's finetune.py follows the LJSpeech dataset format
python "$REPO_DIR/finetune.py" \
  --model_size 1.5B \
  --dataset_path "$DATASET_DIR" \
  --metadata_file "$DATASET_DIR/metadata.csv" \
  --output_dir "$OUTPUT_DIR" \
  --num_epochs 10 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --preserve_voice_cloning \
  --sample_rate 16000

echo ""
echo "=== Fine-tune complete ==="
echo "LoRA checkpoint saved to: $OUTPUT_DIR"
echo ""
echo "To synthesize with the fine-tuned model:"
echo "  python src/voice/synthesize.py --text 'Your text here'"

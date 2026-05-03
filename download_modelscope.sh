#!/bin/bash
# Download Qwen3-ASR models from ModelScope to local disk.

set -euo pipefail

MODELS=(
    "Qwen/Qwen3-ASR-1.7B"
    "Qwen/Qwen3-ASR-0.6B"
    "Qwen/Qwen3-ForcedAligner-0.6B"
)

LOCAL_DIR="${1:-./models}"

echo "Installing modelscope..."
pip install -q modelscope 2>/dev/null || true

echo "Downloading models to: $LOCAL_DIR"
echo ""

for model in "${MODELS[@]}"; do
    echo "  $model ..."
    modelscope download --model "$model" --local_dir "$LOCAL_DIR/$(basename $model)" 2>&1 | tail -1
done

echo ""
echo "Done. Models saved to: $LOCAL_DIR"
echo ""
echo "Launch server with:"
echo "  python asr_server.py --asr-model-path $LOCAL_DIR/Qwen3-ASR-1.7B"

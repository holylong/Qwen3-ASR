#!/bin/bash
# ──────────────────────────────────────────────────────────────────
# Qwen3-ASR WebSocket Server Launcher
# ──────────────────────────────────────────────────────────────────
# Usage:
#   ./start_server.sh                          # HuggingFace, port 8000
#   ./start_server.sh ms                       # ModelScope, port 8000
#   PORT=9000 ./start_server.sh                # custom port
#   MODEL=Qwen/Qwen3-ASR-0.6B ./start_server.sh  # custom model
# ──────────────────────────────────────────────────────────────────

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-ASR-1.7B}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_MEM="${GPU_MEM:-0.5}"
MAX_TOKENS="${MAX_TOKENS:-256}"
CHUNK_SIZE="${CHUNK_SIZE:-1.0}"
UNFIXED_CHUNK="${UNFIXED_CHUNK:-2}"
UNFIXED_TOKEN="${UNFIXED_TOKEN:-5}"
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-}"

MODE="hf"   # default: HuggingFace
if [ "${1:-}" = "ms" ]; then
    MODE="ms"
fi

EXTRA_ARGS=()
if [ "$MODE" = "ms" ]; then
    echo ">>> Using ModelScope for model download"
    EXTRA_ARGS+=(--use-modelscope)
    if [ -n "$MODELSCOPE_CACHE" ]; then
        EXTRA_ARGS+=(--modelscope-cache-dir "$MODELSCOPE_CACHE")
    fi
    # Install modelscope if needed
    pip install -q modelscope 2>/dev/null || true
else
    echo ">>> Using HuggingFace Hub for model loading"
    pip install -q huggingface_hub 2>/dev/null || true
fi

echo "Model      : $MODEL"
echo "Port       : $PORT"
echo "GPU memory : $GPU_MEM"
echo "Max tokens : $MAX_TOKENS"
echo "Chunk size : ${CHUNK_SIZE}s"
echo ""

python asr_server.py \
    --asr-model-path "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-new-tokens "$MAX_TOKENS" \
    --unfixed-chunk-num "$UNFIXED_CHUNK" \
    --unfixed-token-num "$UNFIXED_TOKEN" \
    --chunk-size-sec "$CHUNK_SIZE" \
    "${EXTRA_ARGS[@]}"

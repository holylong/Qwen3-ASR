#!/bin/bash
# ──────────────────────────────────────────────────────────────────
# Qwen3-ASR Server — Simple Launcher (local models)
# ──────────────────────────────────────────────────────────────────
# Usage:
#   ./run.sh                  # default: 1.7B, port 8000
#   ./run.sh 0.6B             # use 0.6B model
#   ./run.sh 1.7B --port 9000 # custom port
#   PORT=9000 ./run.sh        # via env
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

# Model selection
MODEL_NAME="${1:-1.7B}"
MODEL_PATH="$MODELS_DIR/Qwen3-ASR-$MODEL_NAME"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found: $MODEL_PATH"
    echo "Available models:"
    ls -d "$MODELS_DIR"/*/ 2>/dev/null | sed 's|.*/||;s|/$||' | while read d; do echo "  $d"; done
    exit 1
fi

# Config
PORT="${PORT:-8000}"
GPU_MEM="${GPU_MEM:-0.5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TWO_PASS="${TWO_PASS:-}"

echo "========================================"
echo "  Qwen3-ASR Streaming Server"
echo "========================================"
echo "  Model     : $MODEL_PATH"
echo "  Port      : $PORT"
echo "  GPU mem%  : $GPU_MEM"
echo "  Max seq   : $MAX_MODEL_LEN"
echo "  Max tokens: $MAX_TOKENS"
echo "========================================"
echo ""

EXTRA=()
shift 2>/dev/null || true
# pass remaining args through (e.g. --port, --debug)
while [ $# -gt 0 ]; do
    EXTRA+=("$1")
    shift
done

python "$SCRIPT_DIR/asr_server.py" \
    --asr-model-path "$MODEL_PATH" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-new-tokens "$MAX_TOKENS" \
    "${EXTRA[@]}"

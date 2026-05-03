#!/bin/bash
# Start the Qwen3-ASR WebSocket server.
#
# Usage:
#   ./start_server.sh                          # default model, port 8000
#   ./start_server.sh --port 9000              # custom port
#   ./start_server.sh --model /path/to/model   # local model

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-ASR-1.7B}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_MEM="${GPU_MEM:-0.8}"

python asr_server.py \
    --asr-model-path "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-new-tokens 256 \
    --unfixed-chunk-num 2 \
    --unfixed-token-num 5 \
    --chunk-size-sec 1.0


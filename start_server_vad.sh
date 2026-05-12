python asr_server_vad.py \
  --asr-model-path ./models/Qwen3-ASR-0.6B \
  --gpu-memory-utilization 0.85 \
  --max-new-tokens 128 \
  --chunk-size-sec 2.0 \
  --hotwords ./hotwords.json \
  --port 8000 --max-concurrent-requests 8 --debug

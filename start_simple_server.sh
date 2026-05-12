#python asr_server.py --asr-model-path ./models/Qwen3-ASR-1.7B --port 8000
#python asr_server.py --asr-model-path ./models/Qwen3-ASR-1.7B --port 8000 --debug
#python asr_server.py --asr-model-path ./models/Qwen3-ASR-0.6B --port 8000 --debug
#python asr_server.py --asr-model-path ./models/Qwen3-ForcedAligner-0.6B --port 8000 --debug
# python asr_server.py --asr-model-path ./models/Qwen3-ASR-0.6B --port 8000 --debug --save-audio-dir ./debug_audio --save-audio-mode connection

# python asr_server.py --gpu-memory-utilization 0.85 --max-new-tokens 128 --chunk-size-sec 2.0

python asr_server.py \
  --asr-model-path ./models/Qwen3-ASR-0.6B \
  --gpu-memory-utilization 0.85 \
  --max-new-tokens 128 \
  --chunk-size-sec 2.0 \
  --hotwords ./hotwords.json \
  --port 8000 --max-concurrent-requests 8 --debug --save-audio-dir ./debug_audio --save-audio-mode connection
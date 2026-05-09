# Client Latency Optimization

Date: 2026-05-09

## Summary

Optimized `asr_client.py` VAD and audio buffering parameters to reduce perceived recognition latency from ~2.2-3.0s to ~1.0-1.5s.

---

## Changes

### File: `asr_client.py`

| Parameter | Before | After | Effect |
|-----------|--------|-------|--------|
| `PRE_ROLL_SEC` | 1.5s | 0.5s | Less audio buffered before server inference starts (−1.0s) |
| `MIN_SPEECH_FRAMES` | 3 | 2 | VAD triggers faster on speech onset, 750ms→500ms (−0.25s) |
| `SILENCE_DURATION_SEC` | 1.0s | 0.8s | Faster utterance end detection (−0.2s) |
| Pre-roll send | Sequential `await` per chunk | `asyncio.gather` concurrent | Minor reduction in pre-roll flush time |

### Latency Breakdown (perceived from speech onset to first partial result)

| Stage | Before | After |
|-------|--------|-------|
| Mic buffer (fixed) | 250ms | 250ms |
| VAD confirmation | 750ms | 500ms |
| Session handshake (RTT × 2) | ~20-50ms | ~20-50ms |
| Pre-roll replay | 1500ms audio | 500ms audio |
| Server inference (first token) | ~200ms | ~200ms |
| **Total** | **~2.7s** | **~1.5s** |

### New CLI Arguments

```
--min-speech-frames INT    Consecutive speech frames to trigger VAD (default: 2)
--silence-duration SEC     Silence seconds before utterance end (default: 0.8)
```

### Server Optimization (Recommended)

| Parameter | Before | After | Effect |
|-----------|--------|-------|--------|
| `gpu_memory_utilization` | 0.5 | 0.85 | Larger KV cache → higher throughput |
| `max_new_tokens` | 256 | 128 | Less decode overhead per chunk |
| `chunk_size_sec` | 1.0 | 2.0 | Fewer inference calls per utterance |

---

## Usage

```bash
# Server
python asr_server.py \
  --asr-model-path Qwen/Qwen3-ASR-1.7B \
  --gpu-memory-utilization 0.85 \
  --max-new-tokens 128 \
  --chunk-size-sec 2.0 \
  --port 8000

# Client (streaming)
python asr_client.py --url ws://localhost:8000/ws/asr

# Client (two-pass)
python asr_client.py --url ws://localhost:8000/ws/asr --two-pass

# Client (aggressive low latency)
python asr_client.py --url ws://localhost:8000/ws/asr \
  --min-speech-frames 1 --pre-roll-sec 0 --silence-duration 0.5
```

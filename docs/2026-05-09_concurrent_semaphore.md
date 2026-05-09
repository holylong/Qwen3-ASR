# Concurrent Model Access via Semaphore

Date: 2026-05-09

## Summary

Replaced the global `asyncio.Lock()` with `asyncio.Semaphore()` in `asr_server.py` to allow true concurrent model inference across multiple WebSocket sessions. Previously, all model calls were serialized: only one session could run inference at a time, causing other sessions to queue up.

## Changes

### File: `asr_server.py`

| Item | Before | After |
|------|--------|-------|
| Concurrency primitive | `asyncio.Lock()` | `asyncio.Semaphore(4)` |
| Concurrent model calls | 1 (serial) | up to 4 (configurable) |
| `ThreadPoolExecutor` max_workers | 4 | 8 |
| Health endpoint | — | `max_concurrent_requests` field |
| New CLI arg | — | `--max-concurrent-requests` (default 4) |

### Architecture

```
Before (Lock):
  WS1 → [Lock acquired → model call → Lock released]
                                                     WS2 → [Lock acquired → model call → Lock released]

After (Semaphore with N=4):
  WS1 → [Semaphore slot 1 → model call → release]
  WS2 → [Semaphore slot 2 → model call → release]     ← concurrent
  WS3 → [Semaphore slot 3 → model call → release]     ← concurrent
  WS4 → [Semaphore slot 4 → model call → release]     ← concurrent
  WS5 → [wait for slot...]
```

Each session still gets its own `ASRStreamingState`, so there's no state collision between concurrent sessions. The semaphore bounds GPU memory pressure while allowing multiple sessions to make progress simultaneously.

### How vLLM handles concurrent requests

vLLM's internal scheduler uses continuous batching and PagedAttention. When multiple `generate()` calls arrive from different threads, vLLM's engine processes them in parallel — combining their KV caches efficiently. The semaphore at the application level prevents race conditions in the sync `LLM` wrapper while letting the engine do its job.

## Usage

```bash
# Default: 4 concurrent requests
python asr_server.py --asr-model-path Qwen/Qwen3-ASR-1.7B

# High concurrency (more GPU memory needed)
python asr_server.py --asr-model-path Qwen/Qwen3-ASR-1.7B --max-concurrent-requests 8

# Single-threaded (original behavior, lowest risk)
python asr_server.py --asr-model-path Qwen/Qwen3-ASR-1.7B --max-concurrent-requests 1
```

## Tuning

- `--max-concurrent-requests` should not exceed `--executor-max-workers` (hardcoded to 8)
- Increase `--gpu-memory-utilization` (e.g., 0.85) when raising concurrency to give vLLM more KV cache space
- If you see CUDA OOM errors, reduce `--max-concurrent-requests` or `--gpu-memory-utilization`

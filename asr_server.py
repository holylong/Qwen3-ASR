#!/usr/bin/env python3
"""
WebSocket ASR server for Qwen3-ASR with optional 2-pass mode.

Modes:
    streaming (default): Single-pass streaming, low latency.
    two-pass: Pass 1 = streaming (fast preview) + Pass 2 = offline re-transcribe
              on finish (highest accuracy).

Protocol (WebSocket):
    Client -> Server:
        {"type": "start", "mode": "streaming"|"two-pass"}   # begin session
        BINARY: raw float32 PCM 16kHz mono audio chunk
        {"type": "finish"}                                    # end utterance

    Server -> Client:
        {"type": "result", "language": "...", "text": "...",
         "is_partial": true|false, "pass": 1|2}
        {"type": "error", "message": "..."}

Usage:
    # From HuggingFace Hub
    python asr_server.py --asr-model-path Qwen/Qwen3-ASR-1.7B --port 8000

    # From ModelScope (auto-download)
    python asr_server.py --asr-model-path Qwen/Qwen3-ASR-1.7B --use-modelscope --port 8000

    # From ModelScope with custom cache dir
    python asr_server.py --asr-model-path Qwen/Qwen3-ASR-1.7B --use-modelscope \\
        --modelscope-cache-dir ./models --port 8000

Install:
    # HuggingFace backend
    pip install qwen-asr[vllm] fastapi uvicorn

    # ModelScope support (additional)
    pip install modelscope
"""

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, WebSocketException

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("asr_server")

# ---------------------------------------------------------------------------
# Global config / model
# ---------------------------------------------------------------------------
asr_model = None
UNFIXED_CHUNK_NUM = 2
UNFIXED_TOKEN_NUM = 5
CHUNK_SIZE_SEC = 1.0
SESSION_TTL_SEC = 10 * 60

SESSIONS: Dict[str, "Session"] = {}

executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI(title="Qwen3-ASR Streaming Server")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
@dataclass
class Session:
    session_id: str
    ws: WebSocket
    mode: str                # "streaming" | "two-pass"
    state: object
    audio_accum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    @property
    def alive_seconds(self) -> float:
        return time.time() - self.last_seen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gc_sessions() -> None:
    now = time.time()
    dead = [sid for sid, s in SESSIONS.items() if now - s.last_seen > SESSION_TTL_SEC]
    for sid in dead:
        try:
            s = SESSIONS.pop(sid)
        except KeyError:
            pass


def _ensure_still_valid(session_id: str) -> Optional["Session"]:
    _gc_sessions()
    s = SESSIONS.get(session_id)
    if s:
        s.last_seen = time.time()
    return s


async def _send_result(ws: WebSocket, language: str, text: str, is_partial: bool, pass_num: int) -> None:
    try:
        await ws.send_text(json.dumps({
            "type": "result",
            "language": language or "",
            "text": text or "",
            "is_partial": is_partial,
            "pass": pass_num,
        }))
    except Exception:
        pass


async def _send_error(ws: WebSocket, message: str) -> None:
    try:
        await ws.send_text(json.dumps({"type": "error", "message": message}))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Streaming (Pass 1)
# ---------------------------------------------------------------------------
def _streaming_step(pcm: np.ndarray, state) -> None:
    """Blocking call to streaming_transcribe (runs in thread pool)."""
    logger.debug(f"  streaming_transcribe: pcm_len={pcm.shape[0]}, buffer={state.buffer.shape[0]}, "
                 f"chunk_id={state.chunk_id}")
    asr_model.streaming_transcribe(pcm, state)
    logger.debug(f"  streaming_transcribe done: language={state.language!r}, text={state.text[:80]!r}")


def _finish_streaming(state):
    """Blocking finalize call (runs in thread pool)."""
    logger.debug(f"  finish_streaming: accum_audio_len={state.audio_accum.shape[0]}")
    asr_model.finish_streaming_transcribe(state)
    logger.debug(f"  finish_streaming done: language={state.language!r}, text={state.text[:80]!r}")


# ---------------------------------------------------------------------------
# 2-Pass: offline refine (Pass 2)
# ---------------------------------------------------------------------------
def _offline_transcribe(audio: np.ndarray) -> dict:
    """Blocking offline transcribe of accumulated audio (runs in thread pool)."""
    results = asr_model.transcribe(
        audio=(audio, 16000),
        context="",
        language=None,
    )
    r = results[0]
    return {"language": r.language, "text": r.text}


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------
@app.websocket("/ws/asr")
async def websocket_asr(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket connected")

    session_id = None

    try:
        while True:
            raw = await ws.receive()

            if "bytes" in raw:
                # --- Audio chunk (binary) ---
                data = raw["bytes"]
                if len(data) == 0:
                    continue

                s = _ensure_still_valid(session_id)
                if s is None:
                    await _send_error(ws, "No active session. Send 'start' first.")
                    continue

                pcm = np.frombuffer(data, dtype=np.float32).reshape(-1)
                if pcm.size == 0:
                    continue

                logger.debug(f"WS audio chunk: len={pcm.size} samples, session={session_id[:8]}")

                if s.mode == "two-pass":
                    s.audio_accum = np.concatenate([s.audio_accum, pcm], axis=0)

                loop = asyncio.get_running_loop()
                await loop.run_in_executor(executor, _streaming_step, pcm, s.state)
                state = s.state
                await _send_result(
                    ws,
                    language=getattr(state, "language", "") or "",
                    text=getattr(state, "text", "") or "",
                    is_partial=True,
                    pass_num=1,
                )

            elif "text" in raw:
                # --- Control message (JSON) ---
                text = raw["text"]
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    await _send_error(ws, "Invalid JSON")
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "start":
                    mode = msg.get("mode", "streaming")
                    if mode not in ("streaming", "two-pass"):
                        await _send_error(ws, "mode must be 'streaming' or 'two-pass'")
                        continue

                    session_id = uuid.uuid4().hex
                    state = asr_model.init_streaming_state(
                        unfixed_chunk_num=UNFIXED_CHUNK_NUM,
                        unfixed_token_num=UNFIXED_TOKEN_NUM,
                        chunk_size_sec=CHUNK_SIZE_SEC,
                    )
                    SESSIONS[session_id] = Session(
                        session_id=session_id,
                        ws=ws,
                        mode=mode,
                        state=state,
                    )
                    logger.info(f"Session {session_id[:8]} started, mode={mode}")

                    await ws.send_text(json.dumps({
                        "type": "started",
                        "session_id": session_id,
                        "mode": mode,
                    }))

                elif msg_type == "finish":
                    s = _ensure_still_valid(session_id)
                    if s is None:
                        await _send_error(ws, "No active session.")
                        continue

                    loop = asyncio.get_running_loop()

                    # Pass 1 finalize
                    await loop.run_in_executor(executor, _finish_streaming, s.state)
                    state = s.state
                    await _send_result(
                        ws,
                        language=getattr(state, "language", "") or "",
                        text=getattr(state, "text", "") or "",
                        is_partial=False,
                        pass_num=1,
                    )

                    # Pass 2 (offline refine, only in two-pass mode)
                    if s.mode == "two-pass" and s.audio_accum.size > 0:
                        logger.info(f"Session {session_id[:8]}: running pass 2 (offline refine)")
                        result = await loop.run_in_executor(
                            executor, _offline_transcribe, s.audio_accum.copy()
                        )
                        await _send_result(
                            ws,
                            language=result["language"],
                            text=result["text"],
                            is_partial=False,
                            pass_num=2,
                        )

                    SESSIONS.pop(session_id, None)
                    session_id = None
                    logger.info(f"Session finished")

                else:
                    await _send_error(ws, f"Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await _send_error(ws, str(e))
        except Exception:
            pass
    finally:
        if session_id:
            SESSIONS.pop(session_id, None)


# ---------------------------------------------------------------------------
# Model download (ModelScope)
# ---------------------------------------------------------------------------
def _download_from_modelscope(model_id: str, cache_dir: str) -> str:
    logger.info(f"Downloading model from ModelScope: {model_id}")
    try:
        from modelscope import snapshot_download
    except ImportError:
        raise ImportError(
            "modelscope is not installed. Install with: pip install modelscope"
        )
    local_dir = snapshot_download(model_id, cache_dir=cache_dir)
    logger.info(f"Model downloaded to: {local_dir}")
    return local_dir


def _resolve_model_path(model_path: str, use_modelscope: bool,
                        modelscope_cache_dir: str) -> str:
    if not use_modelscope:
        return model_path
    if os.path.isdir(model_path):
        logger.info(f"Using local model directory: {model_path}")
        return model_path
    return _download_from_modelscope(model_path, modelscope_cache_dir)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-ASR WebSocket Server")
    p.add_argument("--asr-model-path", default="Qwen/Qwen3-ASR-1.7B",
                   help="Model name or local path (ModelScope or HuggingFace)")
    p.add_argument("--use-modelscope", action="store_true",
                   help="Download model from ModelScope instead of HuggingFace Hub")
    p.add_argument("--modelscope-cache-dir", default=None,
                   help="ModelScope download cache directory (default: ~/.cache/modelscope)")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=8000, help="Bind port")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                   help="vLLM GPU memory utilization")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Max new tokens for generation")
    p.add_argument("--unfixed-chunk-num", type=int, default=2)
    p.add_argument("--unfixed-token-num", type=int, default=5)
    p.add_argument("--chunk-size-sec", type=float, default=1.0,
                   help="Chunk size in seconds")
    p.add_argument("--debug", action="store_true",
                   help="Enable debug-level logging")
    return p.parse_args()


def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    global asr_model, UNFIXED_CHUNK_NUM, UNFIXED_TOKEN_NUM, CHUNK_SIZE_SEC
    UNFIXED_CHUNK_NUM = args.unfixed_chunk_num
    UNFIXED_TOKEN_NUM = args.unfixed_token_num
    CHUNK_SIZE_SEC = args.chunk_size_sec

    model_path = _resolve_model_path(
        args.asr_model_path,
        use_modelscope=args.use_modelscope,
        modelscope_cache_dir=args.modelscope_cache_dir,
    )

    from qwen_asr import Qwen3ASRModel

    logger.info(f"Loading model from: {model_path}")
    asr_model = Qwen3ASRModel.LLM(
        model=model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=args.max_new_tokens,
    )
    logger.info("Model loaded.")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

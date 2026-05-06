#!/usr/bin/env python3
"""
WebSocket ASR server for Qwen3-ASR with optional 2-pass mode.

Modes:
    streaming (default): Single-pass streaming, low latency.
    two-pass: Pass 1 = streaming (fast preview) + Pass 2 = offline re-transcribe
              on finish (highest accuracy).

Protocol - Qwen3-ASR native (/ws/asr):
    Client -> Server:
        {"type": "start", "mode": "streaming"|"two-pass"}   # begin session
        BINARY: raw float32 PCM 16kHz mono audio chunk
        {"type": "finish"}                                    # end utterance

    Server -> Client:
        {"type": "result", "language": "...", "text": "...",
         "is_partial": true|false, "pass": 1|2}
        {"type": "error", "message": "..."}

Protocol - FunASR compatible (/ws/funasr):
    Client -> Server:
        JSON (first msg): {"mode": "online"|"offline"|"2pass",
            "chunk_size": [5,10,5], "chunk_interval": 10,
            "wav_name": "...", "wav_format": "pcm"|"wav"|"others",
            "is_speaking": true, "hotwords": "...", "itn": true,
            "audio_fs": 16000}
        BINARY: raw int16 PCM or WAV bytes
        JSON (last msg): {"is_speaking": false}

    Server -> Client:
        {"mode": "online"|"offline"|"2pass-online"|"2pass-offline",
         "text": "...", "wav_name": "...", "is_final": true|false,
         "timestamp": "..."}

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
import io
import json
import logging
import os
import struct
import time
import uuid
import wave
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
# FunASR-compatible session
# ---------------------------------------------------------------------------
@dataclass
class FunASRSession:
    session_id: str
    ws: WebSocket
    funasr_mode: str         # "online" | "offline" | "2pass"
    internal_mode: str       # "streaming" | "two-pass" | "offline"
    wav_name: str
    audio_fs: int
    wav_format: str
    itn: bool
    hotwords: str
    context: str
    state: object = None     # ASRStreamingState (for online/2pass)
    audio_accum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    last_sent_text: str = ""  # track last sent text for incremental diff
    started: bool = False
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Audio helpers for FunASR compatibility
# ---------------------------------------------------------------------------
def _pcm_bytes_to_float32(data: bytes, dtype_str: str = "int16") -> np.ndarray:
    """Convert raw PCM bytes to float32 numpy array in [-1, 1]."""
    if dtype_str == "int16":
        pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    elif dtype_str == "float32":
        pcm = np.frombuffer(data, dtype=np.float32)
    else:
        pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm.reshape(-1)


def _parse_wav_bytes(data: bytes) -> np.ndarray:
    """Parse WAV file bytes and return float32 mono 16kHz PCM."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_width == 2:
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        pcm = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels)
        pcm = np.mean(pcm, axis=1).astype(np.float32)

    if sr != 16000:
        try:
            import librosa
            pcm = librosa.resample(pcm, orig_sr=sr, target_sr=16000).astype(np.float32)
        except ImportError:
            ratio = 16000.0 / sr
            n_out = int(len(pcm) * ratio)
            indices = np.linspace(0, len(pcm) - 1, n_out).astype(int)
            pcm = pcm[indices]

    return pcm


def _resample_pcm(pcm: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample float32 PCM to target sample rate."""
    if orig_sr == target_sr or len(pcm) == 0:
        return pcm
    try:
        import librosa
        return librosa.resample(pcm, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    except ImportError:
        ratio = target_sr / orig_sr
        n_out = int(len(pcm) * ratio)
        if n_out == 0:
            return pcm
        indices = np.linspace(0, len(pcm) - 1, n_out).astype(int)
        return pcm[indices]


def _hotwords_to_context(hotwords: str) -> str:
    """Convert FunASR hotwords JSON string to Qwen3-ASR context prompt.

    FunASR hotwords format: {"word1": weight1, "word2": weight2, ...}
    We extract the words and compose a context hint.
    """
    if not hotwords or not hotwords.strip():
        return ""
    try:
        hw_dict = json.loads(hotwords)
        if isinstance(hw_dict, dict):
            words = sorted(hw_dict.keys(), key=lambda k: int(hw_dict[k]) if isinstance(hw_dict[k], (int, float)) else 0, reverse=True)
            return "关注以下词汇: " + ", ".join(words)
    except (json.JSONDecodeError, AttributeError):
        pass
    return ""


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


def _offline_transcribe_with_timestamps(audio: np.ndarray, context: str = "") -> dict:
    """Blocking offline transcribe with optional timestamps (runs in thread pool)."""
    use_timestamps = asr_model.forced_aligner is not None
    try:
        results = asr_model.transcribe(
            audio=(audio, 16000),
            context=context or "",
            language=None,
            return_time_stamps=use_timestamps,
        )
    except Exception:
        results = asr_model.transcribe(
            audio=(audio, 16000),
            context=context or "",
            language=None,
            return_time_stamps=False,
        )
    r = results[0]
    result = {"language": r.language, "text": r.text, "timestamp": ""}
    if use_timestamps and r.time_stamps is not None:
        try:
            ts_str = json.dumps([
                {"text": item.text, "start": item.start_time, "end": item.end_time}
                for item in r.time_stamps.items
            ])
            result["timestamp"] = ts_str
        except Exception:
            pass
    return result


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
# FunASR-compatible WebSocket handler
# ---------------------------------------------------------------------------
FUNASR_SESSIONS: Dict[str, "FunASRSession"] = {}


def _gc_funasr_sessions() -> None:
    now = time.time()
    dead = [sid for sid, s in FUNASR_SESSIONS.items() if now - s.last_seen > SESSION_TTL_SEC]
    for sid in dead:
        FUNASR_SESSIONS.pop(sid, None)


async def _send_funasr_result(ws: WebSocket, mode: str, text: str,
                               wav_name: str, is_final: bool,
                               timestamp: str = "") -> None:
    """Send a result in FunASR protocol format."""
    msg = {
        "mode": mode,
        "text": text or "",
        "wav_name": wav_name or "demo",
        "is_final": is_final,
    }
    if timestamp:
        msg["timestamp"] = timestamp
    try:
        await ws.send_text(json.dumps(msg))
    except Exception:
        pass


@app.websocket("/ws/funasr")
@app.websocket("/")  # FunASR client connects to root path by default
async def websocket_funasr(ws: WebSocket):
    """FunASR-compatible WebSocket endpoint.

    Protocol (client -> server):
        1. First message (JSON): {"mode": "online"|"offline"|"2pass",
           "chunk_size": [5,10,5], "chunk_interval": 10, "wav_name": "...",
           "wav_format": "pcm"|"wav"|"others", "is_speaking": true,
           "hotwords": "...", "itn": true, "audio_fs": 16000}
        2. Binary audio chunks (int16 PCM or WAV bytes)
        3. Final JSON: {"is_speaking": false}

    Protocol (server -> client):
        {"mode": "online"|"offline"|"2pass-online"|"2pass-offline",
         "text": "...", "wav_name": "...", "is_final": true|false,
         "timestamp": "..."}
    """
    await ws.accept(subprotocol="binary")
    logger.info("FunASR WebSocket connected")

    session_id = None
    fsess: Optional[FunASRSession] = None

    try:
        while True:
            raw = await ws.receive()

            if "bytes" in raw:
                # --- Audio chunk (binary) ---
                data = raw["bytes"]
                if len(data) == 0:
                    continue

                if fsess is None or not fsess.started:
                    logger.warning("FunASR: received audio before session init, ignoring")
                    continue

                fsess.last_seen = time.time()

                # Convert audio based on wav_format
                if fsess.wav_format == "wav":
                    pcm = _parse_wav_bytes(data)
                elif fsess.wav_format == "pcm":
                    pcm = _pcm_bytes_to_float32(data, "int16")
                    pcm = _resample_pcm(pcm, fsess.audio_fs, 16000)
                else:
                    # "others" — try WAV first, fallback to int16 PCM
                    try:
                        pcm = _parse_wav_bytes(data)
                    except Exception:
                        pcm = _pcm_bytes_to_float32(data, "int16")
                        pcm = _resample_pcm(pcm, fsess.audio_fs, 16000)

                if pcm.size == 0:
                    continue

                # Accumulate audio for offline / two-pass
                fsess.audio_accum = np.concatenate([fsess.audio_accum, pcm], axis=0)

                if fsess.internal_mode == "offline":
                    # Offline mode: just buffer audio, no streaming results
                    continue

                # Online / 2pass: run streaming step
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(executor, _streaming_step, pcm, fsess.state)
                state = fsess.state
                full_text = getattr(state, "text", "") or ""

                # Compute incremental text (only the new part since last send)
                if full_text.startswith(fsess.last_sent_text):
                    incremental_text = full_text[len(fsess.last_sent_text):]
                else:
                    # Text was revised (e.g., prefix rollback), send full text
                    incremental_text = full_text
                fsess.last_sent_text = full_text

                if fsess.funasr_mode == "2pass":
                    resp_mode = "2pass-online"
                else:
                    resp_mode = "online"

                await _send_funasr_result(
                    ws, mode=resp_mode, text=incremental_text,
                    wav_name=fsess.wav_name, is_final=False,
                )

            elif "text" in raw:
                # --- Control message (JSON) ---
                text_msg = raw["text"]
                try:
                    msg = json.loads(text_msg)
                except json.JSONDecodeError:
                    continue

                is_speaking = msg.get("is_speaking", None)

                # ── First message: session initialization ──
                if is_speaking is True and fsess is None:
                    funasr_mode = msg.get("mode", "2pass")
                    if funasr_mode not in ("online", "offline", "2pass"):
                        funasr_mode = "2pass"

                    wav_name = msg.get("wav_name", "demo")
                    wav_format = msg.get("wav_format", "pcm")
                    audio_fs = int(msg.get("audio_fs", 16000))
                    itn = bool(msg.get("itn", True))
                    hotwords = msg.get("hotwords", "")

                    context = _hotwords_to_context(hotwords)

                    # Map FunASR mode to internal mode
                    if funasr_mode == "online":
                        internal_mode = "streaming"
                    elif funasr_mode == "offline":
                        internal_mode = "offline"
                    else:  # 2pass
                        internal_mode = "two-pass"

                    session_id = uuid.uuid4().hex

                    # Init streaming state (for online/2pass; offline won't use it)
                    state = None
                    if internal_mode != "offline":
                        state = asr_model.init_streaming_state(
                            context=context,
                            unfixed_chunk_num=UNFIXED_CHUNK_NUM,
                            unfixed_token_num=UNFIXED_TOKEN_NUM,
                            chunk_size_sec=CHUNK_SIZE_SEC,
                        )

                    fsess = FunASRSession(
                        session_id=session_id,
                        ws=ws,
                        funasr_mode=funasr_mode,
                        internal_mode=internal_mode,
                        wav_name=wav_name,
                        audio_fs=audio_fs,
                        wav_format=wav_format,
                        itn=itn,
                        hotwords=hotwords,
                        context=context,
                        state=state,
                        started=True,
                    )
                    FUNASR_SESSIONS[session_id] = fsess
                    logger.info(f"FunASR session {session_id[:8]} started, "
                                f"mode={funasr_mode}, wav={wav_name}, fs={audio_fs}")
                    continue

                # ── End of speech: is_speaking=False ──
                if is_speaking is False and fsess is not None:
                    fsess.last_seen = time.time()
                    loop = asyncio.get_running_loop()

                    if fsess.internal_mode == "offline":
                        # Offline mode: transcribe all accumulated audio at once
                        if fsess.audio_accum.size > 0:
                            results = await loop.run_in_executor(
                                executor, _offline_transcribe_with_timestamps,
                                fsess.audio_accum.copy(), fsess.context,
                            )
                            await _send_funasr_result(
                                ws, mode="offline",
                                text=results["text"],
                                wav_name=fsess.wav_name,
                                is_final=True,
                                timestamp=results.get("timestamp", ""),
                            )
                        else:
                            await _send_funasr_result(
                                ws, mode="offline", text="",
                                wav_name=fsess.wav_name, is_final=True,
                            )

                    elif fsess.internal_mode == "streaming":
                        # Online mode: finalize streaming
                        if fsess.state is not None:
                            await loop.run_in_executor(executor, _finish_streaming, fsess.state)
                            state = fsess.state
                            full_text = getattr(state, "text", "") or ""
                            # Final result: send full text
                            await _send_funasr_result(
                                ws, mode="online", text=full_text,
                                wav_name=fsess.wav_name, is_final=True,
                            )

                    elif fsess.internal_mode == "two-pass":
                        # 2pass: finalize streaming (pass 1), then offline refine (pass 2)
                        if fsess.state is not None:
                            await loop.run_in_executor(executor, _finish_streaming, fsess.state)
                            state = fsess.state
                            full_text = getattr(state, "text", "") or ""
                            # Send online final result (full text)
                            await _send_funasr_result(
                                ws, mode="2pass-online", text=full_text,
                                wav_name=fsess.wav_name, is_final=True,
                            )

                        if fsess.audio_accum.size > 0:
                            results = await loop.run_in_executor(
                                executor, _offline_transcribe_with_timestamps,
                                fsess.audio_accum.copy(), fsess.context,
                            )
                            await _send_funasr_result(
                                ws, mode="2pass-offline",
                                text=results["text"],
                                wav_name=fsess.wav_name,
                                is_final=True,
                                timestamp=results.get("timestamp", ""),
                            )

                    # Cleanup
                    FUNASR_SESSIONS.pop(session_id, None)
                    session_id = None
                    fsess = None
                    logger.info("FunASR session finished")

    except WebSocketDisconnect:
        logger.info("FunASR WebSocket disconnected")
    except Exception as e:
        logger.error(f"FunASR WebSocket error: {e}")
    finally:
        if session_id:
            FUNASR_SESSIONS.pop(session_id, None)


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
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5,
                   help="vLLM GPU memory utilization (lower = less KV cache, more free VRAM)")
    p.add_argument("--max-model-len", type=int, default=16384,
                   help="Max sequence length for vLLM (lower = less KV cache VRAM; "
                        "65536=LLM default, 16384=enough for ~10min ASR)")
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
    logger.info(f"  gpu_memory_utilization={args.gpu_memory_utilization}, max_model_len={args.max_model_len}")
    asr_model = Qwen3ASRModel.LLM(
        model=model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
    )
    logger.info("Model loaded.")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

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
         BINARY: raw int16 PCM 16kHz mono audio chunk
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
ASR_CONTEXT = ""              # system prompt built from hotwords config
HOTWORDS_DATA = None          # loaded hotwords dict (wake_words + commands)
UNFIXED_CHUNK_NUM = 2
UNFIXED_TOKEN_NUM = 5
CHUNK_SIZE_SEC = 1.0
SESSION_TTL_SEC = 10 * 60
EXECUTOR_TIMEOUT = 30       # max seconds for a single model inference call
SEND_TIMEOUT = 5            # max seconds for a WebSocket send_text
AUDIO_SAVE_DIR = ""         # if set, save audio to WAV files (see --save-audio-dir)
AUDIO_SAVE_MODE = "session"  # "session" = one file per start/finish; "connection" = one file per WS connect/disconnect

# Concurrent model access is limited by a semaphore. vLLM's sync LLM.generate()
# is NOT fully thread-safe when called concurrently from multiple threads, but
# a bounded semaphore allows several request-worker threads to make progress in
# parallel while keeping GPU memory and internal state safe on most backends.
# Tune --max-concurrent-requests based on GPU VRAM and workload.
MAX_CONCURRENT_REQUESTS = 4
_model_semaphore: asyncio.Semaphore = None  # initialized in main()

SESSIONS: Dict[str, "Session"] = {}

executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="asr_worker")

app = FastAPI(title="Qwen3-ASR Streaming Server")


@app.get("/health")
async def health():
    pending = executor._work_queue.qsize()
    return {
        "status": "ok",
        "sessions": len(SESSIONS),
        "executor_pending": pending,
        "executor_max_workers": executor._max_workers,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
    }


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
def _save_audio_wav(audio: np.ndarray, session_id: str, client_addr: str) -> str:
    """Save accumulated audio to a WAV file. Returns the file path."""
    if not AUDIO_SAVE_DIR or audio.size == 0:
        return ""
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_{client_addr.replace(':', '_')}_{session_id[:8]}.wav"
    path = os.path.join(AUDIO_SAVE_DIR, fname)
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_int16.tobytes())
    logger.info(f"Audio saved: {path} ({audio.size} samples, {audio.size/16000:.1f}s)")
    return path


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


async def _run_in_executor(tag: str, fn, *args):
    """Run fn in the thread pool with a timeout, timing log, and concurrency limit."""
    t0 = time.time()
    async with _model_semaphore:
        t1 = time.time()
        wait_ms = (t1 - t0) * 1000
        if wait_ms > 100:
            logger.info(f"[{tag}] model semaphore: waited {wait_ms:.0f}ms")
        logger.debug(f"[{tag}] executor: submitting... pool_queued={executor._work_queue.qsize()}")
        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(executor, fn, *args),
                timeout=EXECUTOR_TIMEOUT,
            )
            elapsed = (time.time() - t1) * 1000
            logger.debug(f"[{tag}] executor: done in {elapsed:.0f}ms"
                         f" (sem_wait={wait_ms:.0f}ms)")
            return result
        except asyncio.TimeoutError:
            elapsed = (time.time() - t1) * 1000
            logger.error(f"[{tag}] executor: TIMEOUT after {EXECUTOR_TIMEOUT}s"
                          f" (wall={elapsed:.0f}ms, sem_wait={wait_ms:.0f}ms)"
                          f" — model inference hung!")
            raise


async def _send_result(ws: WebSocket, language: str, text: str, is_partial: bool,
                       pass_num: int, hotword_match=None) -> None:
    if not text or not text.strip():
        return
    # Suppress hallucinated output (model echoing context / hotword list)
    if _is_hallucinated(text):
        logger.debug(f"_send_result: suppressed hallucinated output: {text[:80]!r}")
        return
    payload = {
        "type": "result",
        "language": language or "",
        "text": text or "",
        "is_partial": is_partial,
        "pass": pass_num,
    }
    if hotword_match is not None:
        payload["hotword_match"] = hotword_match
    try:
        await asyncio.wait_for(
            ws.send_text(json.dumps(payload)),
            timeout=SEND_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning(f"_send_result: send_text timed out after {SEND_TIMEOUT}s"
                       f" — client not reading?")
    except RuntimeError:
        logger.debug("_send_result: websocket already closed, dropping result")
    except Exception:
        logger.debug("_send_result: send failed, dropping result")


async def _send_error(ws: WebSocket, message: str) -> None:
    try:
        await asyncio.wait_for(
            ws.send_text(json.dumps({"type": "error", "message": message})),
            timeout=SEND_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning(f"_send_error: send_text timed out after {SEND_TIMEOUT}s"
                       f" — error dropped: {message}")
    except RuntimeError:
        logger.debug(f"_send_error: websocket already closed, dropping error: {message}")
    except Exception:
        logger.debug(f"_send_error: send failed, dropping error: {message}")


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
        context=ASR_CONTEXT,
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
    client_host = ws.client.host if ws.client else "?"
    client_port = ws.client.port if ws.client else "?"
    logger.info(f"WebSocket connected: {client_host}:{client_port}")

    session_id = None
    conn_audio = np.zeros(0, dtype=np.float32)

    try:
        while True:
            raw = await ws.receive()
            logger.debug(f"WS recv from {client_host}:{client_port}:"
                         f" type={raw.get('type', '?')},"
                         f" bytes={'bytes' in raw and len(raw.get('bytes', b''))},"
                         f" text={'text' in raw and raw.get('text', '')[:100]!r}")

            # Handle Starlette WebSocket internal disconnect messages
            if raw.get("type") == "websocket.disconnect":
                code = raw.get("code", "?")
                reason = raw.get("reason", "")
                logger.info(f"WebSocket disconnect: {client_host}:{client_port}"
                            f" code={code} reason={reason!r}")
                break

            # Log unexpected message types (neither bytes nor text)
            if "bytes" not in raw and "text" not in raw:
                logger.warning(f"Unrecognized WS message type: {raw.get('type', '?')}"
                               f" from {client_host}:{client_port} keys={list(raw.keys())}")
                continue

            if "bytes" in raw:
                # --- Audio chunk (binary) ---
                data = raw["bytes"]
                if len(data) == 0:
                    continue

                s = _ensure_still_valid(session_id)
                if s is None:
                    logger.warning(f"Binary data without active session"
                                   f" from {client_host}:{client_port}")
                    await _send_error(ws, "No active session. Send 'start' first.")
                    continue

                pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32).reshape(-1) / 32768.0
                if pcm.size == 0:
                    continue

                logger.debug(f"WS audio chunk: len={pcm.size} samples, session={session_id[:8]}")

                s.audio_accum = np.concatenate([s.audio_accum, pcm], axis=0)
                conn_audio = np.concatenate([conn_audio, pcm], axis=0)

                await _run_in_executor("streaming_step", _streaming_step, pcm, s.state)
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
                    logger.warning(f"Invalid JSON from {client_host}:{client_port}:"
                                   f" {text[:200]!r}")
                    await _send_error(ws, "Invalid JSON")
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "start":
                    # Reject duplicate start if a session is already active
                    if session_id is not None:
                        logger.warning(f"Duplicate start rejected from {client_host}:{client_port}"
                                       f" (existing session={session_id[:8]})")
                        await _send_error(ws, "Session already started; finish current first")
                        continue

                    mode = msg.get("mode", "streaming")
                    if mode not in ("streaming", "two-pass"):
                        await _send_error(ws, "mode must be 'streaming' or 'two-pass'")
                        continue

                    session_id = uuid.uuid4().hex
                    async with _model_semaphore:
                        state = asr_model.init_streaming_state(
                            context=ASR_CONTEXT,
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

                    try:
                        await ws.send_text(json.dumps({
                            "type": "started",
                            "session_id": session_id,
                            "mode": mode,
                        }))
                    except (RuntimeError, WebSocketDisconnect) as e:
                        logger.info(f"Client {client_host}:{client_port} disconnected"
                                    f" during session start: {e}")
                        break

                elif msg_type == "finish":
                    s = _ensure_still_valid(session_id)
                    if s is None:
                        await _send_error(ws, "No active session.")
                        continue

                    # Pass 1 finalize
                    await _run_in_executor("finish_streaming", _finish_streaming, s.state)
                    state = s.state
                    text_p1 = getattr(state, "text", "") or ""
                    hw = _match_hotwords(text_p1, HOTWORDS_DATA)
                    await _send_result(
                        ws,
                        language=getattr(state, "language", "") or "",
                        text=text_p1,
                        is_partial=False,
                        pass_num=1,
                        hotword_match=hw,
                    )
                    logger.info(f"Session {session_id[:8]} pass 1 final: {text_p1[:120]}"
                                f"{'  [HOTWORD=' + hw['word'] + ']' if hw else ''}")

                    # Pass 2 (offline refine, only in two-pass mode)
                    if s.mode == "two-pass" and s.audio_accum.size > 0:
                        logger.info(f"Session {session_id[:8]}: running pass 2 (offline refine)")
                        result = await _run_in_executor(
                            "offline_transcribe", _offline_transcribe, s.audio_accum.copy()
                        )
                        hw2 = _match_hotwords(result["text"], HOTWORDS_DATA)
                        await _send_result(
                            ws,
                            language=result["language"],
                            text=result["text"],
                            is_partial=False,
                            pass_num=2,
                            hotword_match=hw2,
                        )
                        logger.info(f"Session {session_id[:8]} pass 2 final: {result['text'][:120]}"
                                    f"{'  [HOTWORD=' + hw2['word'] + ']' if hw2 else ''}")

                    SESSIONS.pop(session_id, None)
                    if AUDIO_SAVE_MODE == "session":
                        _save_audio_wav(s.audio_accum, session_id,
                                        f"{client_host}:{client_port}")
                    session_id = None
                    logger.info(f"Session finished")

                else:
                    logger.warning(f"Unknown message type {msg_type!r}"
                                   f" from {client_host}:{client_port}")
                    await _send_error(ws, f"Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {client_host}:{client_port}"
                    f" (session={session_id[:8] if session_id else 'none'})")
    except WebSocketException:
        logger.info(f"WebSocket exception from {client_host}:{client_port}"
                    f" (session={session_id[:8] if session_id else 'none'})")
    except asyncio.TimeoutError:
        logger.error(f"Executor timeout from {client_host}:{client_port}"
                     f" (session={session_id[:8] if session_id else 'none'})"
                     f" — model inference blocked for >{EXECUTOR_TIMEOUT}s")
    except asyncio.CancelledError:
        logger.info(f"WebSocket handler cancelled: {client_host}:{client_port}")
    except RuntimeError as e:
        logger.error(f"WebSocket runtime error from {client_host}:{client_port}: {e}"
                     f" (session={session_id[:8] if session_id else 'none'})")
    except Exception as e:
        logger.error(f"WebSocket error from {client_host}:{client_port}: {e}"
                     f" (session={session_id[:8] if session_id else 'none'})")
        try:
            await _send_error(ws, str(e))
        except Exception:
            pass
    finally:
        if session_id:
            SESSIONS.pop(session_id, None)
            logger.info(f"Cleaned up session {session_id[:8]} on disconnect")
        if AUDIO_SAVE_MODE == "connection":
            _save_audio_wav(conn_audio, f"conn_{client_host}_{client_port}",
                            f"{client_host}:{client_port}")


# ---------------------------------------------------------------------------
# Model download (ModelScope)
# ---------------------------------------------------------------------------
def _load_hotwords(path: str) -> dict | None:
    """Load hotwords JSON config. Returns dict or None on failure."""
    if not path or not os.path.isfile(path):
        if path:
            logger.warning(f"Hotwords file not found: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load hotwords config: {e}")
        return None
    if not isinstance(data, dict):
        logger.warning("Hotwords config is not a JSON object — ignored")
        return None
    return data


def _build_asr_context(hotwords: dict | None) -> str:
    """Build a SHORT ASR context prompt string from hotwords config.

    IMPORTANT: keep the context under ~30 characters. Long prompts
    cause the model to hallucinate / echo the prompt text when
    there is no real speech input (silence / background noise).
    """
    if not hotwords:
        return ""
    context = hotwords.get("context", "")
    if context:
        return context.strip()[:80]  # hard cap at 80 chars
    # Auto-build a concise hint from key wake words / commands
    wake = hotwords.get("wake_words", [])
    cmds = hotwords.get("commands", [])
    # Show at most 2 wake words + 4 commands as hints
    hints = []
    if wake:
        hints.append("、".join(wake[:2]))
    if cmds:
        hints.append("、".join(cmds[:4]))
    if hints:
        return "控制" + "，如".join(hints)
    return ""
    context = hotwords.get("context", "")
    if context:
        return context.strip()
    wake = hotwords.get("wake_words", [])
    cmds = hotwords.get("commands", [])
    parts = []
    if wake:
        parts.append("唤醒词：" + "、".join(wake))
    if cmds:
        parts.append("可执行指令：" + "、".join(cmds))
    if parts:
        return "你是一个智能语音控制系统。请识别以下语音。" + "。".join(parts) + "。"
    return ""


def _match_hotwords(text: str, hotwords: dict | None) -> dict | None:
    """Check if ASR result matches any wake word or command.

    Returns a dict with matched info, or None if no match.
    Filters out hallucinated output where the model echoes the
    hotword list or context prompt verbatim.
    """
    if not hotwords or not text:
        return None
    text_clean = text.strip().replace(" ", "").replace("，", "").replace("。", "")
    if len(text_clean) < 2:
        return None

    # ── Anti-hallucination: if the output looks like it IS the
    #    hotword list or context, skip matching entirely ──
    wake_words = hotwords.get("wake_words", [])
    commands = hotwords.get("commands", [])
    all_hw = wake_words + commands
    # Count how many hotwords appear in the output
    hit_count = sum(1 for hw in all_hw if hw in text_clean)
    # If ≥ 3 different hotwords appear, it's almost certainly the model
    # echoing the hotword list.  Real speech rarely says 3+ commands at once.
    if hit_count >= 3:
        return None
    # Also check if the output starts with the context itself
    context = hotwords.get("context", "")
    if context and text_clean.startswith(context.replace(" ", "")):
        # If the output is mostly just the context, skip
        if len(text_clean) <= len(context.replace(" ", "")) + 10:
            return None

    # Check wake words first
    for w in wake_words:
        if w in text_clean:
            return {"type": "wake_word", "word": w, "text": text_clean}
    # Check commands
    for c in commands:
        if c in text_clean:
            return {"type": "command", "word": c, "text": text_clean}
    return None


def _is_hallucinated(text: str) -> bool:
    """Check if ASR output is likely a hallucinated echo of the context/hotword list.

    Returns True if the text should be suppressed (not shown to user).
    """
    if not text or not ASR_CONTEXT:
        return False
    t = text.strip().replace(" ", "")
    c = ASR_CONTEXT.strip().replace(" ", "")
    if not c:
        return False
    if t == c or (len(t) >= len(c) and t.startswith(c)):
        return True
    if not HOTWORDS_DATA:
        return False
    all_hw = HOTWORDS_DATA.get("wake_words", []) + HOTWORDS_DATA.get("commands", [])
    hit_count = sum(1 for hw in all_hw if hw in t)
    return hit_count >= 3


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
    p.add_argument("--asr-model-path", default="./models/Qwen3-ASR-0.6B",
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
    p.add_argument("--save-audio-dir", default="",
                   help="Save audio to WAV files in this directory "
                        "(empty = disabled)")
    p.add_argument("--save-audio-mode", default="session",
                   choices=["session", "connection"],
                   help="session: one WAV per start/finish; "
                        "connection: one WAV per WS connect/disconnect")
    p.add_argument("--max-concurrent-requests", type=int, default=4,
                   help="Max simultaneous model inference calls (higher = more concurrency,"
                        " more GPU memory pressure)")
    p.add_argument("--hotwords", default="",
                   help="Path to hotwords JSON config file "
                        "(e.g. hotwords.json). Loads wake words + commands "
                        "and passes them as context prompt to the ASR model.")
    p.add_argument("--debug", action="store_true",
                   help="Enable debug-level logging")
    return p.parse_args()


def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    global asr_model, ASR_CONTEXT, HOTWORDS_DATA, UNFIXED_CHUNK_NUM, UNFIXED_TOKEN_NUM, CHUNK_SIZE_SEC, AUDIO_SAVE_DIR, AUDIO_SAVE_MODE
    global _model_semaphore, MAX_CONCURRENT_REQUESTS
    MAX_CONCURRENT_REQUESTS = args.max_concurrent_requests
    _model_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    UNFIXED_CHUNK_NUM = args.unfixed_chunk_num
    UNFIXED_TOKEN_NUM = args.unfixed_token_num
    CHUNK_SIZE_SEC = args.chunk_size_sec
    AUDIO_SAVE_DIR = args.save_audio_dir
    AUDIO_SAVE_MODE = args.save_audio_mode
    if AUDIO_SAVE_DIR:
        os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)
        logger.info(f"Audio save dir: {AUDIO_SAVE_DIR} (mode={AUDIO_SAVE_MODE})")

    # ── Load hotwords config ─────────────────────────────────────
    HOTWORDS_DATA = _load_hotwords(args.hotwords)
    ASR_CONTEXT = _build_asr_context(HOTWORDS_DATA)
    if HOTWORDS_DATA:
        wake_n = len(HOTWORDS_DATA.get("wake_words", []))
        cmd_n = len(HOTWORDS_DATA.get("commands", []))
        logger.info(f"Hotwords loaded: {wake_n} wake words, {cmd_n} commands"
                    f" from {args.hotwords}")
        logger.info(f"ASR context prompt: {ASR_CONTEXT[:120]}...")
    elif args.hotwords:
        logger.warning(f"Hotwords file specified but could not be loaded: {args.hotwords}")

    model_path = _resolve_model_path(
        args.asr_model_path,
        use_modelscope=args.use_modelscope,
        modelscope_cache_dir=args.modelscope_cache_dir,
    )

    from qwen_asr import Qwen3ASRModel

    logger.info(f"Loading model from: {model_path}")
    logger.info(f"  gpu_memory_utilization={args.gpu_memory_utilization}, max_model_len={args.max_model_len}")
    logger.info(f"  max_concurrent_requests={args.max_concurrent_requests}, executor_workers=8")
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

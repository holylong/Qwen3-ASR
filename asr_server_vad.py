#!/usr/bin/env python3
"""
WebSocket ASR server with Silero VAD for automatic sentence segmentation.

Server-side VAD detects speech boundaries and automatically manages ASR
sessions. Clients simply stream continuous audio — no VAD or session
management needed on the client side.

Architecture:
    Client streams raw int16 PCM 16kHz mono audio continuously via WebSocket.
    Server runs Silero VAD frame-by-frame to detect speech boundaries.
    When speech starts → ASR session created, audio streamed through model.
    When speech ends  → ASR finalized, optional pass-2 offline refinement.
    Partial results sent during speech, final results sent after silence.

Protocol (WebSocket):
    Client -> Server (optional):
        {"type": "set_mode", "mode": "streaming"|"two-pass"}
    Server -> Client:
        {"type": "result", "language": "...", "text": "...",
         "is_partial": true|false, "pass": 1|2}
        {"type": "vad_state", "state": "speaking"|"listening"}
        {"type": "error", "message": "..."}

Usage:
    python asr_server_vad.py --asr-model-path ./models/Qwen3-ASR-0.6B --port 8000

Install:
    pip install qwen-asr[vllm] fastapi uvicorn torch
    # Silero VAD is auto-downloaded via torch.hub on first run
"""

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, WebSocketException

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("asr_server_vad")

# ---------------------------------------------------------------------------
# Global config / model
# ---------------------------------------------------------------------------
asr_model = None
vad_model = None
ASR_CONTEXT = ""              # system prompt built from hotwords config
HOTWORDS_DATA = None          # loaded hotwords dict (wake_words + commands)
UNFIXED_CHUNK_NUM = 2
UNFIXED_TOKEN_NUM = 5
CHUNK_SIZE_SEC = 1.0
SESSION_TTL_SEC = 10 * 60
EXECUTOR_TIMEOUT = 30
SEND_TIMEOUT = 5
AUDIO_SAVE_DIR = ""
AUDIO_SAVE_MODE = "session"

MAX_CONCURRENT_REQUESTS = 4
_model_semaphore: asyncio.Semaphore = None

executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="asr_worker")
app = FastAPI(title="Qwen3-ASR Streaming Server (VAD)")

# VAD tuning knobs
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 300
VAD_SPEECH_PAD_MS = 100
VAD_PRE_ROLL_CHUNKS = 2          # number of 0.25s chunks buffered before speech onset
VAD_FRAME_SIZE = 512             # 32 ms @ 16 kHz
VAD_SAMPLE_RATE = 16000
MAX_UTTERANCE_SEC = 30           # force-finish if speech exceeds this


@app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(SESSIONS) if "SESSIONS" in dir() else 0}


# ---------------------------------------------------------------------------
# ASR Session (one per utterance)
# ---------------------------------------------------------------------------
@dataclass
class Session:
    session_id: str
    ws: WebSocket
    mode: str
    state: object
    audio_accum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_audio_wav(audio: np.ndarray, session_id: str, client_addr: str) -> str:
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
    logger.info(f"Audio saved: {path} ({audio.size / 16000:.1f}s)")
    return path


async def _run_in_executor(tag: str, fn, *args):
    """Run fn in the thread pool with a timeout and concurrency limit."""
    t0 = time.time()
    async with _model_semaphore:
        t1 = time.time()
        wait_ms = (t1 - t0) * 1000
        if wait_ms > 100:
            logger.info(f"[{tag}] model semaphore: waited {wait_ms:.0f}ms")
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
            logger.error(f"[{tag}] executor: TIMEOUT after {EXECUTOR_TIMEOUT}s")
            raise


async def _send_result(ws: WebSocket, language: str, text: str,
                       is_partial: bool, pass_num: int,
                       hotword_match=None) -> None:
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
        logger.warning(f"_send_result: send_text timed out after {SEND_TIMEOUT}s")
    except RuntimeError:
        logger.debug("_send_result: websocket already closed")
    except Exception:
        logger.debug("_send_result: send failed")


async def _send_error(ws: WebSocket, message: str) -> None:
    try:
        await asyncio.wait_for(
            ws.send_text(json.dumps({"type": "error", "message": message})),
            timeout=SEND_TIMEOUT,
        )
    except Exception:
        pass


async def _send_vad_state(ws: WebSocket, state: str) -> None:
    try:
        await asyncio.wait_for(
            ws.send_text(json.dumps({"type": "vad_state", "state": state})),
            timeout=SEND_TIMEOUT,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# ASR model calls (blocking, run in executor)
# ---------------------------------------------------------------------------
def _streaming_step(pcm: np.ndarray, state) -> None:
    asr_model.streaming_transcribe(pcm, state)


def _finish_streaming(state):
    asr_model.finish_streaming_transcribe(state)


def _offline_transcribe(audio: np.ndarray) -> dict:
    results = asr_model.transcribe(
        audio=(audio, 16000),
        context=ASR_CONTEXT,
        language=None,
    )
    r = results[0]
    return {"language": r.language, "text": r.text}


def _init_asr_state():
    return asr_model.init_streaming_state(
        context=ASR_CONTEXT,
        unfixed_chunk_num=UNFIXED_CHUNK_NUM,
        unfixed_token_num=UNFIXED_TOKEN_NUM,
        chunk_size_sec=CHUNK_SIZE_SEC,
    )


# ---------------------------------------------------------------------------
# Silero VAD engine
# ---------------------------------------------------------------------------
class VADEngine:
    """Streaming VAD using raw Silero model output.

    Implements the same hysteresis state machine as VADIterator but calls
    the model directly, avoiding dependency on silero-vad's utility class
    (whose API may change between versions).

    Processes audio frame-by-frame and emits events when speech starts or
    ends. Maintains a pre-roll buffer so that the ASR model receives audio
    covering the speech onset.
    """

    def __init__(self):
        self.model = vad_model
        self.threshold = VAD_THRESHOLD
        self.sampling_rate = VAD_SAMPLE_RATE
        self.min_silence_samples = int(
            VAD_SAMPLE_RATE * VAD_MIN_SILENCE_MS / 1000
        )
        self.speech_pad_samples = int(
            VAD_SAMPLE_RATE * VAD_SPEECH_PAD_MS / 1000
        )
        self.pre_roll_chunks: deque = deque(maxlen=VAD_PRE_ROLL_CHUNKS)
        self._temp_buffer = np.zeros(0, dtype=np.float32)
        self.reset()

    def reset(self):
        self._triggered = False
        self._temp_end = 0
        self._current_sample = 0
        self.pre_roll_chunks.clear()
        self._temp_buffer = np.zeros(0, dtype=np.float32)

    def process(self, chunk: np.ndarray):
        """Process one audio chunk (float32, any length).

        Returns a list of (event_type, data) tuples:
          ("start", [chunks])   – speech started; data = pre-roll chunks
          ("data", chunk)       – speech audio to feed to ASR
          ("end", None)         – speech ended
          ("silence", chunk)    – non-speech (ignored by caller)
        """
        events: list = []

        # Store chunk in pre-roll *before* VAD so the start event includes
        # the chunk that triggered the onset.
        self.pre_roll_chunks.append(chunk.copy())

        # Concatenate leftover from previous call for frame alignment.
        audio = np.concatenate([self._temp_buffer, chunk])
        self._temp_buffer = np.zeros(0, dtype=np.float32)

        triggered_this_call = False

        num_frames = len(audio) // VAD_FRAME_SIZE
        for i in range(num_frames):
            begin = i * VAD_FRAME_SIZE
            end = begin + VAD_FRAME_SIZE
            frame = audio[begin:end]
            t = torch.from_numpy(frame.copy()).float()

            with torch.no_grad():
                speech_prob = self.model(t, self.sampling_rate).item()

            self._current_sample += VAD_FRAME_SIZE

            # ── Speech onset detection ──
            if speech_prob >= self.threshold and self._temp_end != 0:
                self._temp_end = 0

            if speech_prob >= self.threshold and not self._triggered:
                self._triggered = True
                triggered_this_call = True
                events.append(("start", list(self.pre_roll_chunks)))

            # ── Speech end detection ──
            if speech_prob < (self.threshold - 0.15) and self._triggered:
                if self._temp_end == 0:
                    self._temp_end = self._current_sample
                gap = self._current_sample - self._temp_end
                if gap >= self.min_silence_samples:
                    self._triggered = False
                    self._temp_end = 0
                    events.append(("end", None))

        # Preserve incomplete frame for next call.
        leftover = len(audio) % VAD_FRAME_SIZE
        if leftover > 0:
            self._temp_buffer = audio[-leftover:].copy()

        # Emit data / silence for this chunk.
        if self._triggered and not triggered_this_call:
            events.append(("data", chunk))
        elif not self._triggered:
            events.append(("silence", chunk))

        return events


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------
@app.websocket("/ws/asr")
async def websocket_asr(ws: WebSocket):
    await ws.accept()
    client_addr = f"{ws.client.host}:{ws.client.port}" if ws.client else "?:?"

    logger.info(f"WebSocket connected: {client_addr}")

    mode: str = "streaming"
    vad_engine: VADEngine = VADEngine()
    asr_session: Optional[Session] = None
    conn_audio = np.zeros(0, dtype=np.float32)
    utt_count = 0  # per-connection utterance counter

    try:
        while True:
            raw = await ws.receive()

            # --- Internal disconnect message (Starlette) ---
            if raw.get("type") == "websocket.disconnect":
                logger.info(f"WebSocket disconnect: {client_addr}")
                break

            if "bytes" not in raw and "text" not in raw:
                logger.warning(f"Unrecognized WS message from {client_addr}:"
                               f" type={raw.get('type', '?')} keys={list(raw.keys())}")
                continue

            # ====================================================
            # Binary  →  audio chunk
            # ====================================================
            if "bytes" in raw:
                data = raw["bytes"]
                if len(data) == 0:
                    continue

                pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32).reshape(-1) / 32768.0
                if pcm.size == 0:
                    continue

                conn_audio = np.concatenate([conn_audio, pcm], axis=0)

                # ---- VAD processing ---------------------------------
                events = vad_engine.process(pcm)

                for ev_type, ev_data in events:

                    if ev_type == "start":
                        utt_count += 1
                        logger.info(f"[{client_addr}] Speech #{utt_count} started"
                                    f" (mode={mode})")
                        await _send_vad_state(ws, "speaking")

                        # Create ASR streaming state
                        state = await _run_in_executor("init_state", _init_asr_state)
                        session_id = uuid.uuid4().hex
                        asr_session = Session(
                            session_id=session_id,
                            ws=ws,
                            mode=mode,
                            state=state,
                        )
                        logger.debug(f"  session={session_id[:8]}")

                        # Feed pre-roll chunks into the ASR model.
                        # Each chunk goes through streaming_transcribe so
                        # the model's internal state is primed with the
                        # speech onset audio.
                        pre_roll_chunks = ev_data  # list of float32 arrays
                        for i, pr_chunk in enumerate(pre_roll_chunks):
                            if pr_chunk.size == 0:
                                continue
                            await _run_in_executor(
                                f"pre_roll[{i}]", _streaming_step,
                                pr_chunk, asr_session.state,
                            )
                            asr_session.audio_accum = np.concatenate(
                                [asr_session.audio_accum, pr_chunk]
                            )

                        # Send earliest partial result after pre-roll
                        st = asr_session.state
                        text = getattr(st, "text", "") or ""
                        if text.strip():
                            await _send_result(
                                ws,
                                getattr(st, "language", "") or "",
                                text,
                                is_partial=True,
                                pass_num=1,
                            )

                    elif ev_type == "data":
                        if asr_session is None:
                            continue
                        # Check max utterance duration
                        elapsed = time.time() - asr_session.created_at
                        if elapsed > MAX_UTTERANCE_SEC:
                            logger.warning(f"[{client_addr}] Utterance #{utt_count}"
                                           f" exceeded {MAX_UTTERANCE_SEC}s — forcing finish")
                            await _finish_asr_session(asr_session, ws, utt_count)
                            asr_session = None
                            vad_engine.reset()
                            continue

                        await _run_in_executor(
                            f"streaming", _streaming_step,
                            ev_data, asr_session.state,
                        )
                        asr_session.audio_accum = np.concatenate(
                            [asr_session.audio_accum, ev_data]
                        )
                        st = asr_session.state
                        text = getattr(st, "text", "") or ""
                        if text.strip():
                            await _send_result(
                                ws,
                                getattr(st, "language", "") or "",
                                text,
                                is_partial=True,
                                pass_num=1,
                            )

                    elif ev_type == "end":
                        if asr_session is None:
                            continue
                        logger.info(f"[{client_addr}] Speech #{utt_count} ended")
                        await _send_vad_state(ws, "listening")
                        await _finish_asr_session(asr_session, ws, utt_count)
                        asr_session = None
                        vad_engine.reset()

            # ====================================================
            # Text  →  control message (JSON)
            # ====================================================
            elif "text" in raw:
                try:
                    msg = json.loads(raw["text"])
                except json.JSONDecodeError:
                    await _send_error(ws, "Invalid JSON")
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "set_mode":
                    new_mode = msg.get("mode", "streaming")
                    if new_mode in ("streaming", "two-pass"):
                        mode = new_mode
                        logger.info(f"[{client_addr}] Mode set → {mode}")
                        try:
                            await ws.send_text(json.dumps({"type": "mode_set", "mode": mode}))
                        except Exception:
                            pass
                    else:
                        await _send_error(ws, "mode must be 'streaming' or 'two-pass'")

                elif msg_type == "ping":
                    try:
                        await ws.send_text(json.dumps({"type": "pong"}))
                    except Exception:
                        pass

                else:
                    logger.debug(f"[{client_addr}] Unknown msg type: {msg_type!r}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {client_addr}")
    except WebSocketException as e:
        logger.info(f"WebSocket exception: {client_addr}: {e}")
    except asyncio.TimeoutError:
        logger.error(f"Executor timeout from {client_addr}")
    except asyncio.CancelledError:
        logger.info(f"Handler cancelled: {client_addr}")
    except RuntimeError as e:
        logger.error(f"Runtime error from {client_addr}: {e}")
    except Exception as e:
        logger.error(f"WebSocket error from {client_addr}: {type(e).__name__}: {e}")
        try:
            await _send_error(ws, str(e))
        except Exception:
            pass
    finally:
        # Clean up in-flight ASR session if any
        if asr_session is not None:
            try:
                await _finish_asr_session(asr_session, ws, utt_count)
            except Exception:
                pass
        if AUDIO_SAVE_MODE == "connection" and conn_audio.size > 0:
            _save_audio_wav(conn_audio, f"conn_{client_addr.replace(':', '_')}",
                            client_addr)
        logger.info(f"Connection closed: {client_addr}")


async def _finish_asr_session(session: Session, ws: WebSocket, utt_num: int):
    """Finalize an ASR streaming session and send results."""
    try:
        # Pass 1 – finalize streaming
        await _run_in_executor("finish_streaming", _finish_streaming, session.state)
        st = session.state
        text_p1 = getattr(st, "text", "") or ""
        # In two-pass mode, only send hotword_match with the higher-quality
        # pass-2 result to avoid duplicate match lines.
        is_two_pass = (session.mode == "two-pass")
        await _send_result(
            ws,
            getattr(st, "language", "") or "",
            text_p1,
            is_partial=False,
            pass_num=1,
            hotword_match=_match_hotwords(text_p1, HOTWORDS_DATA) if not is_two_pass else None,
        )
        logger.info(f"  Pass 1 final [{session.session_id[:8]}]: {text_p1[:120]}")

        # Pass 2 – offline refine (two-pass mode only)
        if is_two_pass and session.audio_accum.size > 0:
            logger.info(f"  Running pass 2 (offline refine)")
            result = await _run_in_executor(
                "offline_transcribe",
                _offline_transcribe,
                session.audio_accum.copy(),
            )
            hw2 = _match_hotwords(result["text"], HOTWORDS_DATA)
            await _send_result(
                ws,
                result["language"],
                result["text"],
                is_partial=False,
                pass_num=2,
                hotword_match=hw2,
            )
            logger.info(f"  Pass 2 final [{session.session_id[:8]}]: {result['text'][:120]}"
                        f"{'  [HOTWORD=' + hw2['word'] + ']' if hw2 else ''}")

        if AUDIO_SAVE_MODE == "session":
            _save_audio_wav(session.audio_accum, session.session_id, "")

    except Exception as e:
        logger.error(f"  finish_asr_session error: {e}")


# ---------------------------------------------------------------------------
# Model download helpers (ModelScope)
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
    # Validate minimal structure
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

    return None


def _is_hallucinated(text: str) -> bool:
    """Check if ASR output is likely a hallucinated echo of the context/hotword list.

    Returns True if the text should be suppressed (not shown to user).
    """
    if not text or not ASR_CONTEXT:
        return False
    # Exact match or near-match of the context
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

    # Check wake words first
    for w in wake_words:
        if w in text_clean:
            return {"type": "wake_word", "word": w, "text": text_clean}
    # Check commands
    for c in commands:
        if c in text_clean:
            return {"type": "command", "word": c, "text": text_clean}
    return None


def _download_from_modelscope(model_id: str, cache_dir: str) -> str:
    logger.info(f"Downloading model from ModelScope: {model_id}")
    try:
        from modelscope import snapshot_download
    except ImportError:
        raise ImportError("modelscope not installed. pip install modelscope")
    local_dir = snapshot_download(model_id, cache_dir=cache_dir)
    logger.info(f"Model downloaded to: {local_dir}")
    return local_dir


def _resolve_model_path(model_path: str, use_modelscope: bool,
                        modelscope_cache_dir: str) -> str:
    if not use_modelscope:
        return model_path
    if os.path.isdir(model_path):
        return model_path
    return _download_from_modelscope(model_path, modelscope_cache_dir)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-ASR WebSocket Server with Silero VAD")
    p.add_argument("--asr-model-path", default="./models/Qwen3-ASR-0.6B")
    p.add_argument("--use-modelscope", action="store_true")
    p.add_argument("--modelscope-cache-dir", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--unfixed-chunk-num", type=int, default=2)
    p.add_argument("--unfixed-token-num", type=int, default=5)
    p.add_argument("--chunk-size-sec", type=float, default=1.0)
    p.add_argument("--save-audio-dir", default="")
    p.add_argument("--save-audio-mode", default="session",
                   choices=["session", "connection"])
    p.add_argument("--max-concurrent-requests", type=int, default=4)
    p.add_argument("--vad-threshold", type=float, default=0.5,
                   help="Silero VAD speech probability threshold (0.0–1.0)")
    p.add_argument("--vad-min-silence-ms", type=int, default=300,
                   help="Minimum silence before sentence end (ms)")
    p.add_argument("--vad-speech-pad-ms", type=int, default=100,
                   help="Padding around speech segments (ms)")
    p.add_argument("--vad-pre-roll-chunks", type=int, default=2,
                   help="Number of 0.25 s audio chunks buffered before speech onset")
    p.add_argument("--max-utterance-sec", type=int, default=30,
                   help="Maximum utterance duration before forced finish")
    p.add_argument("--hotwords", default="",
                   help="Path to hotwords JSON config file "
                        "(e.g. hotwords.json). Loads wake words + commands "
                        "and passes them as context prompt to the ASR model.")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    global asr_model, vad_model, ASR_CONTEXT, HOTWORDS_DATA
    global UNFIXED_CHUNK_NUM, UNFIXED_TOKEN_NUM, CHUNK_SIZE_SEC
    global AUDIO_SAVE_DIR, AUDIO_SAVE_MODE
    global _model_semaphore, MAX_CONCURRENT_REQUESTS
    global VAD_THRESHOLD, VAD_MIN_SILENCE_MS, VAD_SPEECH_PAD_MS, VAD_PRE_ROLL_CHUNKS
    global MAX_UTTERANCE_SEC

    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    MAX_CONCURRENT_REQUESTS = args.max_concurrent_requests
    _model_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    UNFIXED_CHUNK_NUM = args.unfixed_chunk_num
    UNFIXED_TOKEN_NUM = args.unfixed_token_num
    CHUNK_SIZE_SEC = args.chunk_size_sec
    AUDIO_SAVE_DIR = args.save_audio_dir
    AUDIO_SAVE_MODE = args.save_audio_mode
    VAD_THRESHOLD = args.vad_threshold
    VAD_MIN_SILENCE_MS = args.vad_min_silence_ms
    VAD_SPEECH_PAD_MS = args.vad_speech_pad_ms
    VAD_PRE_ROLL_CHUNKS = args.vad_pre_roll_chunks
    MAX_UTTERANCE_SEC = args.max_utterance_sec

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

    # ── Load Silero VAD ──────────────────────────────────────────
    # Cache silero-vad model files locally under ./models so it is
    # downloaded only once (not on every server start).
    _vad_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(_vad_cache_dir, exist_ok=True)
    torch.hub.set_dir(_vad_cache_dir)
    logger.info(f"Loading Silero VAD model...  (cache dir: {_vad_cache_dir})")
    vad_model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
    )
    vad_model.eval()
    logger.info("Silero VAD model loaded (CPU)")

    # ── Load ASR model ───────────────────────────────────────────
    model_path = _resolve_model_path(
        args.asr_model_path,
        use_modelscope=args.use_modelscope,
        modelscope_cache_dir=args.modelscope_cache_dir,
    )

    from qwen_asr import Qwen3ASRModel

    logger.info(f"Loading ASR model from: {model_path}")
    logger.info(f"  gpu_memory_utilization={args.gpu_memory_utilization},"
                f" max_model_len={args.max_model_len}")
    logger.info(f"  max_concurrent_requests={args.max_concurrent_requests}")
    asr_model = Qwen3ASRModel.LLM(
        model=model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
    )
    logger.info("ASR model loaded.")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

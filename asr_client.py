#!/usr/bin/env python3
"""
Real-time ASR client with microphone input.

Connects to the Qwen3-ASR WebSocket server and performs streaming speech
recognition. Supports single-pass (streaming) and two-pass (streaming +
offline refinement) modes.

Architecture: concurrent send + receive tasks.
  - Receive task: continuously reads WebSocket, pushes results to queue
  - Send task: drains audio, runs VAD, sends audio, handles coordination

Usage:
    python asr_client.py --url ws://localhost:8000/ws/asr
    python asr_client.py --url ws://localhost:8000/ws/asr --two-pass
    python asr_client.py --url ws://localhost:8000/ws/asr --verbose
    python asr_client.py --list-devices

Install:
    pip install sounddevice websockets numpy
"""

import argparse
import asyncio
import json
import os
import queue
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.25
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

VAD_THRESHOLD = 0.015
SILENCE_DURATION_SEC = 1.0
MIN_SPEECH_FRAMES = 3
PRE_ROLL_SEC = 1.5

STATE_IDLE = "idle"
STATE_SPEAKING = "speaking"


# ──────────────────────────────────────────────
# VAD
# ──────────────────────────────────────────────
class VADState:
    def __init__(self, threshold=VAD_THRESHOLD,
                 silence_sec=SILENCE_DURATION_SEC,
                 min_speech=MIN_SPEECH_FRAMES):
        self.threshold = threshold
        self.silence_frames = max(1, int(silence_sec / CHUNK_DURATION))
        self.min_speech = max(1, min_speech)
        self.speech_frames = 0
        self.silent_frames = 0
        self.state = STATE_IDLE

    def update(self, is_speech: bool) -> str:
        if self.state == STATE_IDLE:
            if is_speech:
                self.speech_frames += 1
                if self.speech_frames >= self.min_speech:
                    self.state = STATE_SPEAKING
                    self.silent_frames = 0
                    return STATE_SPEAKING
            else:
                self.speech_frames = max(0, self.speech_frames - 1)
            return STATE_IDLE
        else:
            if is_speech:
                self.silent_frames = 0
            else:
                self.silent_frames += 1
                if self.silent_frames >= self.silence_frames:
                    self.state = STATE_IDLE
                    self.speech_frames = 0
                    return STATE_IDLE
            return STATE_SPEAKING


# ──────────────────────────────────────────────
# Terminal UI
# ──────────────────────────────────────────────
def _tw() -> int:
    try:
        return os.get_terminal_size().columns
    except Exception:
        return 80


def _trim(s, w=0):
    if w <= 0:
        w = _tw()
    a = max(20, w - 10)
    return s if len(s) <= a else s[:a] + "..."


class TUI:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def _c(self):
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def log(self, *a):
        if not self.verbose:
            return
        self._c()
        sys.stdout.write("\r  \033[90m" + " ".join(str(x) for x in a) + "\033[0m\n")
        sys.stdout.flush()

    def partial(self, text, lang=""):
        self._c()
        l = f"[{lang}] " if lang else ""
        sys.stdout.write(f"\r  \033[33m●\033[0m {l}{_trim(text)}")
        sys.stdout.flush()

    def final(self, text, lang="", p=1, elapsed=0.0):
        self._c()
        pl = f"(P{p}) " if p >= 2 else ""
        l = f"[{lang}] " if lang else ""
        t = f"  \033[90m{elapsed:.1f}s\033[0m" if elapsed > 0 else ""
        sys.stdout.write(f"\r  \033[32m✔\033[0m {pl}{l}{text}{t}\n")
        sys.stdout.flush()

    def err(self, m):
        self._c()
        sys.stdout.write(f"\r  \033[31m✖\033[0m {m}\n")
        sys.stdout.flush()

    def status(self, m):
        self._c()
        sys.stdout.write(f"\r  \033[90m… {m}\033[0m")
        sys.stdout.flush()


# ──────────────────────────────────────────────
# Shared state between send & receive tasks
# ──────────────────────────────────────────────
@dataclass
class SessionCtx:
    session_id: str = ""
    mode: str = "streaming"
    active: bool = False
    start_ok: asyncio.Event = None  # set by receiver when 'started' msg arrives

    def __post_init__(self):
        if self.start_ok is None:
            self.start_ok = asyncio.Event()


# ──────────────────────────────────────────────
# Audio capture helpers
# ──────────────────────────────────────────────
class MicCapture:
    def __init__(self, device=None, blocksize=CHUNK_SAMPLES, record: bool = False):
        self.q: queue.Queue = queue.Queue()
        self.stream = None
        self.device = device
        self.blocksize = blocksize
        self.record = record
        self._recording: list = []  # accumulated audio for saving

    def _cb(self, indata, frames, ti, status):
        data = indata[:, 0].copy()
        self.q.put(data)
        if self.record:
            self._recording.append(data)

    def start(self):
        import sounddevice as sd
        self._recording.clear()
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype=np.float32,
            callback=self._cb, blocksize=self.blocksize, device=self.device,
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def drain_one(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def save_wav(self, filepath: str):
        if not self.record or not self._recording:
            return
        import soundfile as sf
        audio = np.concatenate(self._recording)
        sf.write(filepath, audio, SAMPLE_RATE)
        return filepath


# ──────────────────────────────────────────────
# Receive task (continuous background reader)
# ──────────────────────────────────────────────
async def recv_loop(ws, result_q: asyncio.Queue, ctx: SessionCtx, tui: TUI, stop: threading.Event):
    while not stop.is_set():
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        except Exception:
            break
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue

        t = msg.get("type", "")
        if t == "started":
            ctx.session_id = msg.get("session_id", "")
            ctx.mode = msg.get("mode", "streaming")
            ctx.active = True
            ctx.start_ok.set()
        elif t == "result":
            is_partial = msg.get("is_partial", False)
            if is_partial:
                tui.partial(msg.get("text", ""), msg.get("language", ""))
            # final results are displayed by send_loop with timing
        elif t == "error":
            tui.err(msg.get("message", "Server error"))
        elif tui.verbose:
            tui.log("  recv:", t)

        await result_q.put(msg)


# ──────────────────────────────────────────────
# Send task (drains audio, runs VAD, sends)
# ──────────────────────────────────────────────
async def send_loop(ws, mic: MicCapture, result_q: asyncio.Queue,
                     ctx: SessionCtx, tui: TUI, args, stop: threading.Event):
    vad = VADState(threshold=args.vad_threshold)
    prev_state = STATE_IDLE
    pre_roll_max = max(1, int(args.pre_roll_sec / CHUNK_DURATION))
    pre_roll: deque = deque(maxlen=pre_roll_max)

    flush_deadline = 0.0
    final_results_pending = 0

    while not stop.is_set():
        chunk = mic.drain_one()

        # Drain result queue (non-blocking, just consume stale messages)
        while not result_q.empty():
            try:
                result_q.get_nowait()
            except asyncio.QueueEmpty:
                break

        if chunk is None:
            await asyncio.sleep(0.02)
            continue

        if chunk.size == 0:
            continue

        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        is_speech = rms > args.vad_threshold
        new_state = vad.update(is_speech)

        if args.verbose:
            tui.log(f"VAD: rms={rms:.4f} thr={args.vad_threshold} "
                    f"speech={is_speech} state={vad.state} "
                    f"sf={vad.speech_frames}/{vad.min_speech} pre={len(pre_roll)}")

        # ── IDLE → SPEAKING ──
        if prev_state == STATE_IDLE and new_state == STATE_SPEAKING:
            ctx.start_ok.clear()
            ctx.active = False
            mode = "two-pass" if args.two_pass else "streaming"
            tui.status("Speech detected — starting session...")
            await ws.send(json.dumps({"type": "start", "mode": mode}))
            try:
                await asyncio.wait_for(ctx.start_ok.wait(), timeout=10)
            except asyncio.TimeoutError:
                tui.err("Start timed out")
                prev_state = new_state
                continue

            sid = ctx.session_id[:8] if ctx.session_id else "?"
            tui.status(f"[{sid}] Processing...")

            if pre_roll:
                for c in list(pre_roll):
                    await ws.send((c * 32767).clip(-32768, 32767).astype(np.int16).tobytes())
                if args.verbose:
                    tui.log(f"  pre-roll: sent {len(pre_roll)} chunks "
                            f"({len(pre_roll) * CHUNK_DURATION:.1f}s)")
                pre_roll.clear()

        # ── SPEAKING: send chunk ──
        if new_state == STATE_SPEAKING and ctx.active:
            await ws.send((chunk * 32767).clip(-32768, 32767).astype(np.int16).tobytes())

        # ── SPEAKING → IDLE ──
        if prev_state == STATE_SPEAKING and new_state == STATE_IDLE:
            if ctx.active:
                tui.status("Speech ended — finishing...")
                finish_time = asyncio.get_event_loop().time()
                await ws.send(json.dumps({"type": "finish"}))
                await _wait_final_results(result_q, tui, timeout=8.0,
                                          finish_time=finish_time)
                ctx.active = False
            tui.status("Listening...")

        # Append to pre-roll AFTER all send operations
        pre_roll.append(chunk.copy())
        prev_state = new_state

    # Shutdown
    if ctx.active:
        tui.status("Stopping — finishing...")
        try:
            finish_time = asyncio.get_event_loop().time()
            await ws.send(json.dumps({"type": "finish"}))
            await _wait_final_results(result_q, tui, timeout=4.0,
                                      finish_time=finish_time)
        except Exception:
            pass


async def _wait_final_results(result_q: asyncio.Queue, tui: TUI, timeout: float,
                              finish_time: float = 0.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            msg = await asyncio.wait_for(result_q.get(), timeout=min(2.0, remaining))
        except asyncio.TimeoutError:
            break
        if msg.get("type") == "result" and not msg.get("is_partial", True):
            elapsed = asyncio.get_event_loop().time() - finish_time if finish_time > 0 else 0.0
            tui.final(msg.get("text", ""), msg.get("language", ""),
                      msg.get("pass", 1), elapsed)
            if msg.get("pass", 1) >= 2:
                break
        elif msg.get("type") == "error":
            tui.err(msg.get("message", "Server error"))
            break


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
async def list_audio_devices():
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. pip install sounddevice")
        return
    print("Available audio input devices:\n")
    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            sr = dev.get("default_samplerate", 0)
            print(f"  [{i}] {dev['name']}  (in:{dev['max_input_channels']}ch, "
                  f"SR:{int(sr) if sr else '?'}Hz)")
    print()


async def run_client(args):
    import websockets
    tui = TUI(verbose=args.verbose)

    device = None
    if args.device is not None:
        try:
            device = int(args.device)
        except ValueError:
            device = args.device

    mic = MicCapture(device=device, record=args.save_audio)
    ctx = SessionCtx()
    result_q: asyncio.Queue = asyncio.Queue()
    stop_flag = threading.Event()

    def on_sig(sig, frame):
        stop_flag.set()
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    tui.status("Connecting...")
    try:
        async with websockets.connect(
            args.url, ping_interval=20, ping_timeout=10, max_size=2**24,
        ) as ws:
            mic.start()
            tui.status("Connected. Speak now... (Ctrl+C to stop)")

            recv_task = asyncio.create_task(recv_loop(ws, result_q, ctx, tui, stop_flag))
            send_task = asyncio.create_task(send_loop(ws, mic, result_q, ctx, tui, args, stop_flag))

            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()

            mic.stop()
            await ws.close()
            tui._c()
            print("Done.")

    except websockets.exceptions.InvalidURI:
        tui.err(f"Invalid URL: {args.url}")
    except websockets.exceptions.ConnectionClosed as e:
        tui.err(f"Connection closed: {e}")
    except OSError as e:
        tui.err(f"Connection failed: {e}")
    except ImportError as e:
        tui.err(str(e))
    except Exception as e:
        tui.err(f"{type(e).__name__}: {e}")
    finally:
        mic.stop()
        if args.save_audio:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = mic.save_wav(args.save_audio if args.save_audio != "1"
                                else f"recording_{ts}.wav")
            if path:
                tui._c()
                print(f"Audio saved: {path}")
            else:
                tui._c()
        else:
            tui._c()


def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time ASR client (Qwen3-ASR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python asr_client.py --url ws://localhost:8000/ws/asr
  python asr_client.py --url ws://localhost:8000/ws/asr --two-pass
  python asr_client.py --url ws://localhost:8000/ws/asr --verbose
  python asr_client.py --list-devices""",
    )
    p.add_argument("--url", default="ws://localhost:8000/ws/asr")
    p.add_argument("--two-pass", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--vad-threshold", type=float, default=VAD_THRESHOLD)
    p.add_argument("--pre-roll-sec", type=float, default=PRE_ROLL_SEC)
    p.add_argument("--save-audio", nargs="?", const="1", default=None,
                   help="Save captured audio to WAV file (optional: path)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list_devices:
        asyncio.run(list_audio_devices())
        return
    try:
        asyncio.run(run_client(args))
    except KeyboardInterrupt:
        sys.stdout.write("\r\033[K"); sys.stdout.flush()
        print("Interrupted.")


if __name__ == "__main__":
    main()

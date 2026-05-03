#!/usr/bin/env python3
"""
Real-time ASR client with microphone input.

Connects to the Qwen3-ASR WebSocket server and performs streaming speech
recognition. Supports single-pass (streaming) and two-pass (streaming +
offline refinement) modes.

Audio is captured at 16kHz mono and processed through energy-based VAD.
Speech segments are streamed to the server in real-time.

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
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.25
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

# VAD
VAD_THRESHOLD = 0.015
SILENCE_DURATION_SEC = 1.0
MIN_SPEECH_FRAMES = 3            # warm-up before streaming triggers
PRE_ROLL_SEC = 1.5               # pre-roll buffer (must be >= warm-up time)

STATE_IDLE = "idle"
STATE_SPEAKING = "speaking"


class VADState:
    """Energy-based Voice Activity Detection."""

    def __init__(self, threshold: float = VAD_THRESHOLD,
                 silence_sec: float = SILENCE_DURATION_SEC,
                 min_speech_frames: int = MIN_SPEECH_FRAMES):
        self.threshold = threshold
        self.silence_frames = max(1, int(silence_sec / CHUNK_DURATION))
        self.min_speech_frames = max(1, min_speech_frames)
        self.speech_frames = 0
        self.silent_frames = 0
        self.state = STATE_IDLE

    def is_speech(self, chunk: np.ndarray) -> bool:
        if chunk.size == 0:
            return False
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        return rms > self.threshold

    def update(self, is_speech_frame: bool) -> str:
        if self.state == STATE_IDLE:
            if is_speech_frame:
                self.speech_frames += 1
                if self.speech_frames >= self.min_speech_frames:
                    self.state = STATE_SPEAKING
                    self.silent_frames = 0
                    return STATE_SPEAKING
            else:
                self.speech_frames = max(0, self.speech_frames - 1)
            return STATE_IDLE
        else:
            if is_speech_frame:
                self.silent_frames = 0
            else:
                self.silent_frames += 1
                if self.silent_frames >= self.silence_frames:
                    self.state = STATE_IDLE
                    self.speech_frames = 0
                    return STATE_IDLE
            return STATE_SPEAKING


def _term_width() -> int:
    try:
        return os.get_terminal_size().columns
    except Exception:
        return 80


def _trim(text: str, width: int = 0) -> str:
    if width <= 0:
        width = _term_width()
    avail = max(20, width - 10)
    return text if len(text) <= avail else text[:avail] + "..."


class TerminalUI:
    """Minimal terminal display with optional verbose mode."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _clear(self):
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def echo(self, *args):
        if not self.verbose:
            return
        self._clear()
        sys.stdout.write("\r  \033[90m" + " ".join(str(a) for a in args) + "\033[0m\n")
        sys.stdout.flush()

    def partial(self, text: str, language: str = ""):
        self._clear()
        lang = f"[{language}] " if language else ""
        sys.stdout.write(f"\r  \033[33m●\033[0m {lang}{_trim(text)}")
        sys.stdout.flush()

    def final(self, text: str, language: str = "", pass_num: int = 1):
        self._clear()
        p_label = f"(P{pass_num}) " if pass_num >= 2 else ""
        lang = f"[{language}] " if language else ""
        sys.stdout.write(f"\r  \033[32m✔\033[0m {p_label}{lang}{text}\n")
        sys.stdout.flush()

    def error(self, msg: str):
        self._clear()
        sys.stdout.write(f"\r  \033[31m✖\033[0m {msg}\n")
        sys.stdout.flush()

    def status(self, msg: str):
        self._clear()
        sys.stdout.write(f"\r  \033[90m… {msg}\033[0m")
        sys.stdout.flush()


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
            print(f"  [{i}] {dev['name']}  "
                  f"(in: {dev['max_input_channels']}ch, "
                  f"SR: {int(sr) if sr else '?'}Hz)")
    print()


async def run_client(args: argparse.Namespace):
    import sounddevice as sd
    import websockets

    ui = TerminalUI(verbose=args.verbose)
    audio_queue: queue.Queue = queue.Queue()
    stop_flag = threading.Event()
    stream = None

    def cb(indata, frames, time_info, status):
        if status:
            ui.echo(f"Audio status: {status}")
        audio_queue.put(indata[:, 0].copy())

    device = None
    if args.device is not None:
        try:
            device = int(args.device)
        except ValueError:
            device = args.device

    def start_mic():
        nonlocal stream
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype=np.float32,
            callback=cb, blocksize=CHUNK_SAMPLES, device=device,
        )
        stream.start()

    def stop_mic():
        nonlocal stream
        if stream:
            stream.stop()
            stream.close()
            stream = None

    running = True
    def on_signal(sig, frame):
        nonlocal running
        running = False
        stop_flag.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # Pending audio chunks from sounddevice callback (not yet VAD-processed).
    pending_chunks: list = []

    def drain_one_chunk() -> Optional[np.ndarray]:
        """Return a single 250ms chunk if available; None otherwise."""
        nonlocal pending_chunks
        try:
            while True:
                pending_chunks.append(audio_queue.get_nowait())
        except queue.Empty:
            pass
        if pending_chunks:
            return pending_chunks.pop(0)
        return None

    async def read_pending_results(ws, wait: float = 0.03) -> bool:
        """Read any pending messages from server."""
        got_any = False
        while True:
            try:
                msg_text = await asyncio.wait_for(ws.recv(), timeout=wait)
            except asyncio.TimeoutError:
                break
            except websockets.exceptions.ConnectionClosed:
                break
            try:
                msg = json.loads(msg_text)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "result":
                is_partial = msg.get("is_partial", False)
                p = msg.get("pass", 1)
                lang = msg.get("language", "")
                txt = msg.get("text", "")
                if is_partial:
                    ui.partial(txt, lang)
                else:
                    ui.final(txt, lang, p)
                got_any = True
            elif msg.get("type") == "error":
                ui.error(msg.get("message", "Server error"))
                got_any = True
            elif args.verbose:
                ui.echo("  server:", msg.get("type", "?"))
        return got_any

    async def finish_session(ws) -> None:
        """Send finish, wait for final results."""
        await ws.send(json.dumps({"type": "finish"}))
        deadline = asyncio.get_event_loop().time() + 8.0
        got_p1 = False
        while asyncio.get_event_loop().time() < deadline:
            try:
                remaining = max(0.1, deadline - asyncio.get_event_loop().time())
                msg_text = await asyncio.wait_for(ws.recv(), timeout=min(2.0, remaining))
            except asyncio.TimeoutError:
                break
            except websockets.exceptions.ConnectionClosed:
                break
            try:
                msg = json.loads(msg_text)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "result" and not msg.get("is_partial", True):
                p = msg.get("pass", 1)
                ui.final(msg.get("text", ""), msg.get("language", ""), p)
                if p == 1:
                    got_p1 = True
                if p >= 2:
                    break
            elif msg.get("type") == "error":
                ui.error(msg.get("message", "Server error"))
                break

    # ──── main loop ────
    ui.status("Connecting...")
    try:
        async with websockets.connect(
            args.url,
            ping_interval=20,
            ping_timeout=10,
            max_size=2**24,
        ) as ws:

            mode = "two-pass" if args.two_pass else "streaming"
            vad = VADState(threshold=args.vad_threshold)
            prev_state = STATE_IDLE
            have_session = False
            pre_roll_max = max(1, int(args.pre_roll_sec / CHUNK_DURATION))
            pre_roll_buf: deque = deque(maxlen=pre_roll_max)

            start_mic()
            ui.status("Connected. Speak now... (Ctrl+C to stop)")

            while running and not stop_flag.is_set():
                chunk = drain_one_chunk()

                # Always drain server results
                await read_pending_results(ws, wait=0.01)

                if chunk is None:
                    await asyncio.sleep(0.02)
                    continue

                # Compute speech flag
                if chunk.size > 0:
                    rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
                    is_speech_frame = rms > args.vad_threshold
                    if args.verbose:
                        ui.echo(f"VAD: rms={rms:.4f} thr={args.vad_threshold} speech={is_speech_frame} "
                                f"state={vad.state} sf={vad.speech_frames}/{vad.min_speech_frames}")
                else:
                    is_speech_frame = False

                new_state = vad.update(is_speech_frame)

                # ---- IDLE → SPEAKING: flush pre-roll, then start session ----
                if prev_state == STATE_IDLE and new_state == STATE_SPEAKING:
                    ui.status("Speech detected — starting session...")
                    await ws.send(json.dumps({"type": "start", "mode": mode}))
                    resp = await asyncio.wait_for(ws.recv(), timeout=10)
                    r = json.loads(resp)
                    if r.get("type") != "started":
                        ui.error(f"Start failed: {r}")
                        prev_state = new_state
                        continue
                    have_session = True
                    sid = r.get("session_id", "?")[:8]
                    ui.status(f"[{sid}] Processing...")

                    # Flush pre-roll buffer so first words aren't lost
                    if pre_roll_buf:
                        for buf_chunk in list(pre_roll_buf):
                            await ws.send(buf_chunk.tobytes())
                        if args.verbose:
                            ui.echo(f"  pre-roll: sent {len(pre_roll_buf)} chunks "
                                    f"({len(pre_roll_buf) * CHUNK_DURATION:.1f}s)")
                        pre_roll_buf.clear()
                    await read_pending_results(ws, wait=0.05)

                # ---- SPEAKING: send chunk ----
                if new_state == STATE_SPEAKING and have_session:
                    await ws.send(chunk.tobytes())
                    await read_pending_results(ws, wait=0.05)

                # ---- SPEAKING → IDLE ----
                if prev_state == STATE_SPEAKING and new_state == STATE_IDLE:
                    if have_session:
                        ui.status("Speech ended — finishing...")
                        await finish_session(ws)
                        have_session = False
                        # drain remaining
                        await read_pending_results(ws, wait=0.1)
                    ui.status("Listening...")

                # Append to pre-roll AFTER sending (avoids double-send on transition)
                pre_roll_buf.append(chunk.copy())

                prev_state = new_state

            # Shutdown
            if have_session:
                ui.status("Stopping — finishing last session...")
                try:
                    await ws.send(json.dumps({"type": "finish"}))
                    await asyncio.wait_for(read_pending_results(ws, wait=0.3), timeout=4.0)
                except Exception:
                    pass

            stop_mic()
            ui._clear()
            print("Done.")

    except websockets.exceptions.InvalidURI:
        ui.error(f"Invalid URL: {args.url}")
    except websockets.exceptions.ConnectionClosed as e:
        ui.error(f"Connection closed: {e}")
    except OSError as e:
        ui.error(f"Connection failed: {e}")
    except ImportError as e:
        ui.error(str(e))
    except Exception as e:
        ui.error(f"{type(e).__name__}: {e}")
    finally:
        stop_mic()
        ui._clear()


def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time ASR client with microphone (Qwen3-ASR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python asr_client.py --url ws://localhost:8000/ws/asr
  python asr_client.py --url ws://localhost:8000/ws/asr --two-pass
  python asr_client.py --url ws://localhost:8000/ws/asr --verbose
  python asr_client.py --url ws://localhost:8000/ws/asr --vad-threshold 0.01
  python asr_client.py --list-devices
        """,
    )
    p.add_argument("--url", default="ws://localhost:8000/ws/asr", help="WebSocket server URL")
    p.add_argument("--two-pass", action="store_true", help="Enable 2-pass mode")
    p.add_argument("--device", default=None, help="Audio input device index or name")
    p.add_argument("--list-devices", action="store_true", help="List devices and exit")
    p.add_argument("--vad-threshold", type=float, default=VAD_THRESHOLD,
                   help=f"VAD RMS threshold (default: {VAD_THRESHOLD})")
    p.add_argument("--pre-roll-sec", type=float, default=PRE_ROLL_SEC,
                   help=f"Pre-roll buffer seconds (default: {PRE_ROLL_SEC})")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print VAD RMS values and debug info")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list_devices:
        asyncio.run(list_audio_devices())
        return
    try:
        asyncio.run(run_client(args))
    except KeyboardInterrupt:
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
        print("Interrupted.")


if __name__ == "__main__":
    main()

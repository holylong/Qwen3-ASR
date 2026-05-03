#!/usr/bin/env python3
"""
Real-time ASR client with microphone input.

Connects to the Qwen3-ASR WebSocket server and performs streaming speech
recognition. Supports single-pass (streaming) and two-pass (streaming +
offline refinement) modes.

Each speech segment (detected by energy-based VAD) starts a new ASR session.
Results are displayed in real-time in the terminal.

Usage:
    python asr_client.py --url ws://localhost:8000/ws/asr
    python asr_client.py --url ws://localhost:8000/ws/asr --two-pass
    python asr_client.py --url ws://localhost:8000/ws/asr --list-devices

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
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

VAD_THRESHOLD = 0.02
SILENCE_DURATION_SEC = 0.8
MIN_SPEECH_FRAMES = 4

STATE_IDLE = "idle"
STATE_SPEAKING = "speaking"


class VADState:
    """Energy-based Voice Activity Detection."""

    def __init__(self, threshold: float = VAD_THRESHOLD,
                 silence_sec: float = SILENCE_DURATION_SEC):
        self.threshold = threshold
        self.silence_frames = int(silence_sec / CHUNK_DURATION)
        self.speech_frames = 0
        self.silent_frames = 0
        self.state = STATE_IDLE

    def is_speech(self, chunk: np.ndarray) -> bool:
        if chunk.size == 0:
            return False
        rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))
        return rms > self.threshold

    def update(self, chunk: np.ndarray) -> str:
        speech = self.is_speech(chunk)
        if self.state == STATE_IDLE:
            if speech:
                self.speech_frames += 1
                if self.speech_frames >= MIN_SPEECH_FRAMES:
                    self.state = STATE_SPEAKING
                    self.silent_frames = 0
                    return STATE_SPEAKING
            else:
                self.speech_frames = 0
            return STATE_IDLE
        else:  # STATE_SPEAKING
            if speech:
                self.silent_frames = 0
            else:
                self.silent_frames += 1
                if self.silent_frames >= self.silence_frames:
                    self.state = STATE_IDLE
                    self.speech_frames = 0
                    return STATE_IDLE
            return STATE_SPEAKING


def _terminal_width() -> int:
    try:
        return os.get_terminal_size().columns
    except Exception:
        return 80


def _trim(text: str, width: int = 0) -> str:
    if width <= 0:
        width = _terminal_width()
    avail = max(20, width - 8)
    return text if len(text) <= avail else text[:avail] + "..."


class TerminalUI:
    """Minimal terminal display."""

    def _clear(self):
        sys.stdout.write("\r\033[K")
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
            print(f"  [{i}] {dev['name']}  (in: {dev['max_input_channels']}ch, "
                  f"SR: {int(sr) if sr else '?'}Hz)")
    print()


async def run_client(args: argparse.Namespace):
    import sounddevice as sd
    import websockets

    ui = TerminalUI()
    audio_queue: queue.Queue = queue.Queue()
    stop_flag = threading.Event()
    stream = None

    def cb(indata, frames, time_info, status):
        if status:
            print(f"\nAudio: {status}", file=sys.stderr)
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

    async def drain_audio():
        """Drain audio_queue into a numpy chunk, or return None if empty."""
        parts = []
        try:
            while True:
                parts.append(audio_queue.get_nowait())
        except queue.Empty:
            pass
        if not parts:
            return None
        return np.concatenate(parts)

    async def read_server_results(ws, wait: float = 0.05):
        """Read any pending result messages from server."""
        while True:
            try:
                msg_text = await asyncio.wait_for(ws.recv(), timeout=wait)
            except asyncio.TimeoutError:
                return
            except websockets.exceptions.ConnectionClosed:
                return
            try:
                msg = json.loads(msg_text)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "result":
                is_partial = msg.get("is_partial", False)
                if is_partial:
                    ui.partial(msg.get("text", ""), msg.get("language", ""))
                else:
                    ui.final(msg.get("text", ""), msg.get("language", ""),
                             msg.get("pass", 1))
            elif msg.get("type") == "error":
                ui.error(msg.get("message", "Server error"))

    async def finish_session(ws, ui):
        """Send finish, then wait for final results (pass 1 + optional pass 2)."""
        await ws.send(json.dumps({"type": "finish"}))
        seen_p1 = False
        start = asyncio.get_event_loop().time()
        while True:
            timeout = max(0.1, 5.0 - (asyncio.get_event_loop().time() - start))
            if timeout <= 0:
                break
            try:
                msg_text = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                break
            except websockets.exceptions.ConnectionClosed:
                break
            try:
                msg = json.loads(msg_text)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "result" and not msg.get("is_partial"):
                ui.final(msg.get("text", ""), msg.get("language", ""),
                         msg.get("pass", 1))
                pass_num = msg.get("pass", 1)
                if pass_num == 1:
                    seen_p1 = True
                if pass_num >= 2:
                    break
            elif msg.get("type") == "error":
                ui.error(msg.get("message", "Server error"))
                break
        return seen_p1

    # ---- main connection loop ----
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

            start_mic()
            ui.status("Connected. Speak now... (Ctrl+C to stop)")

            while running and not stop_flag.is_set():
                chunk = await drain_audio()
                await read_server_results(ws, wait=0.02)

                if chunk is None:
                    await asyncio.sleep(0.05)
                    continue

                new_state = vad.update(chunk)

                # --- IDLE -> SPEAKING: start a new session ---
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

                # --- SPEAKING: send audio chunk ---
                if new_state == STATE_SPEAKING and have_session:
                    await ws.send(chunk.tobytes())
                    await read_server_results(ws, wait=0.05)

                # --- SPEAKING -> IDLE: finish session ---
                if prev_state == STATE_SPEAKING and new_state == STATE_IDLE:
                    if have_session:
                        ui.status("Speech ended — finishing...")
                        await finish_session(ws, ui)
                        have_session = False
                    ui.status("Listening...")

                prev_state = new_state

            if have_session:
                ui.status("Stopping — finishing last session...")
                try:
                    await ws.send(json.dumps({"type": "finish"}))
                    await asyncio.wait_for(read_server_results(ws, wait=0.3), timeout=3.0)
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
  python asr_client.py --url ws://localhost:8000/ws/asr --device 1
  python asr_client.py --list-devices
        """,
    )
    p.add_argument("--url", default="ws://localhost:8000/ws/asr", help="WebSocket server URL")
    p.add_argument("--two-pass", action="store_true", help="Enable 2-pass mode (streaming + offline refine)")
    p.add_argument("--device", default=None, help="Audio input device index or name")
    p.add_argument("--list-devices", action="store_true", help="List audio input devices and exit")
    p.add_argument("--vad-threshold", type=float, default=VAD_THRESHOLD, help="VAD RMS energy threshold (default: 0.02)")
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

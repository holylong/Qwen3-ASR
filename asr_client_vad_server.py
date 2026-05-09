#!/usr/bin/env python3
"""
Real-time ASR client — no VAD on client side.

Streams microphone audio continuously to the Qwen3-ASR server, which
handles VAD-based sentence segmentation with Silero VAD.

Usage:
    python asr_client_vad_server.py --url ws://localhost:8000/ws/asr
    python asr_client_vad_server.py --url ws://localhost:8000/ws/asr --two-pass
    python asr_client_vad_server.py --url ws://localhost:8000/ws/asr --verbose
    python asr_client_vad_server.py --list-devices

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

import numpy as np

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.25
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)


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

    def vad_state(self, state):
        if state == "speaking":
            self.status("Speaking... (VAD)")
        else:
            self.status("Listening... (VAD)")


# ──────────────────────────────────────────────
# Audio capture
# ──────────────────────────────────────────────
class MicCapture:
    def __init__(self, device=None, blocksize=CHUNK_SAMPLES, record: bool = False):
        self.q: queue.Queue = queue.Queue()
        self.stream = None
        self.device = device
        self.blocksize = blocksize
        self.record = record
        self._recording: list = []

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
# WebSocket send / receive
# ──────────────────────────────────────────────
async def recv_loop(ws, result_q: asyncio.Queue, tui: TUI, stop: threading.Event):
    """Continuously read messages from the server."""
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

        if t == "result":
            is_partial = msg.get("is_partial", False)
            text = msg.get("text", "")
            lang = msg.get("language", "")
            pass_num = msg.get("pass", 1)
            if is_partial and text.strip():
                tui.partial(text, lang)
            elif not is_partial and text.strip():
                tui.final(text, lang, pass_num)

        elif t == "vad_state":
            state = msg.get("state", "")
            tui.vad_state(state)

        elif t == "error":
            tui.err(msg.get("message", "Server error"))

        elif t == "pong":
            pass  # keepalive

        elif t == "mode_set":
            if tui.verbose:
                tui.log(f"Mode confirmed: {msg.get('mode', '?')}")

        elif tui.verbose:
            tui.log("  recv:", t)

        await result_q.put(msg)


async def send_loop(ws, mic: MicCapture, tui: TUI, stop: threading.Event):
    """Continuously capture audio and send to server. No VAD, no session mgmt."""
    while not stop.is_set():
        chunk = mic.drain_one()
        if chunk is None:
            await asyncio.sleep(0.02)
            continue
        if chunk.size == 0:
            continue

        payload = (chunk * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        try:
            await ws.send(payload)
        except Exception as e:
            if tui.verbose:
                tui.log(f"Send error: {e}")
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
            # Tell server which mode to use
            mode = "two-pass" if args.two_pass else "streaming"
            await ws.send(json.dumps({"type": "set_mode", "mode": mode}))
            tui.log(f"Mode set to: {mode}")

            mic.start()
            tui.status("Connected. Speak now... (Ctrl+C to stop)")
            tui.vad_state("listening")

            recv_task = asyncio.create_task(
                recv_loop(ws, result_q, tui, stop_flag)
            )
            send_task = asyncio.create_task(
                send_loop(ws, mic, tui, stop_flag)
            )

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


def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time ASR client — server-side VAD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python asr_client_vad_server.py --url ws://localhost:8000/ws/asr
  python asr_client_vad_server.py --url ws://localhost:8000/ws/asr --two-pass
  python asr_client_vad_server.py --url ws://localhost:8000/ws/asr --verbose
  python asr_client_vad_server.py --list-devices""",
    )
    p.add_argument("--url", default="ws://localhost:8000/ws/asr")
    p.add_argument("--two-pass", action="store_true",
                   help="Enable two-pass mode (streaming + offline refine)")
    p.add_argument("--device", default=None,
                   help="Audio input device index or name")
    p.add_argument("--list-devices", action="store_true",
                   help="List audio input devices and exit")
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

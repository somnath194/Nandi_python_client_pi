"""
nandi_assistant_pi.py — Real-time conversational voice assistant (Raspberry Pi)

Pipeline:
  1. 🔇 IDLE      — Porcupine listens for "Hey Nandi" wake word
  2. 🎙️ LISTENING — arecord | WebRTC VAD → OpenAI Whisper transcribes
  3. 🤖 THINKING  — Sends transcription to LLM WebSocket (streaming)
  4. 🔊 SPEAKING  — Plays LLM response via OpenAI TTS + ffplay

NOTE: PyAudio is NOT used for input or output — it was segfaulting on Pi.
      Audio capture  → arecord  (alsa-utils)
      Beep playback  → aplay    (alsa-utils)
      TTS playback   → ffplay   (ffmpeg)

Hotkeys (in terminal):
  Ctrl+R  — restart pipeline instantly (unstick if frozen)
  Ctrl+C  — quit

Requirements:
    pip install pvporcupine pvrecorder webrtcvad openai websockets python-dotenv
    sudo apt install ffmpeg alsa-utils -y
    pip install webrtcvad-wheels   # if webrtcvad won't compile

.env (one level up from this script):
    PICOVOICE_ACCESS_KEY=...
    OPENAI_API_KEY=...
"""

import array
import asyncio
import collections
import ctypes
import io
import json
import math
import os
import select
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
import tty
import wave
from pathlib import Path
from typing import Optional

import pvporcupine
import webrtcvad
from pvrecorder import PvRecorder
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
import websockets

# ── Suppress ALSA error spam ──────────────────────────────────────────────────
try:
    _asound = ctypes.cdll.LoadLibrary("libasound.so.2")
    _ALSA_ERROR_FUNC = ctypes.CFUNCTYPE(
        None, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
    )
    _asound.snd_lib_error_set_handler(_ALSA_ERROR_FUNC(lambda *_: None))
except Exception:
    pass


# ── Env / Config ──────────────────────────────────────────────────────────────
dotenv_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)

ACCESS_KEY     = os.getenv("PICOVOICE_ACCESS_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_PATH     = os.getenv(
    "WAKE_WORD_MODEL_PATH",
    "/home/somi/Documents/nandi_python_client/data/hay-nandi_en_raspberry-pi_v4_0_0.ppn",
)

LLM_WS_URI = os.getenv("LLM_WS_URI")
SESSION_ID = os.getenv("LLM_SESSION_ID", "nandi-session")

# ── Audio device config ────────────────────────────────────────────────────────
# For PvRecorder device index run:
#   python -c "from pvrecorder import PvRecorder; [print(i,d) for i,d in enumerate(PvRecorder.get_available_devices())]"
PVRECORDER_DEVICE_INDEX : int = -1   # -1 = default

# For arecord device run:  arecord -l
# Use format "hw:X,Y"  e.g. "hw:1,0" for card 1 device 0
# "default" works in most cases
ARECORD_DEVICE : str = "default"

# For aplay output device: aplay -l
# "default" works in most cases
APLAY_DEVICE   : str = "default"

# ── WebRTC VAD + Whisper STT settings ─────────────────────────────────────────
STT_SAMPLE_RATE    : int   = 16_000
VAD_FRAME_MS       : int   = 30
VAD_FRAME_SAMPLES  : int   = int(STT_SAMPLE_RATE * VAD_FRAME_MS / 1000)   # 480
VAD_FRAME_BYTES    : int   = VAD_FRAME_SAMPLES * 2                         # 960
VAD_AGGRESSIVENESS : int   = 3   # 0=gentle … 3=most strict

SPEECH_TRIGGER_FRAMES  : int   = 6     # voiced frames to start (~180 ms)
SILENCE_TRIGGER_FRAMES : int   = 20    # silent frames to stop  (~600 ms)
MAX_SEGMENT_SEC        : float = 25.0  # hard cap per utterance
MIN_SPEECH_FRAMES      : int   = 15    # discard clips shorter than ~450 ms
PRE_ROLL_FRAMES        : int   = 10    # 300 ms prepended before speech onset

WHISPER_LANGUAGE : Optional[str] = "en"   # None = auto-detect

# ── Follow-up window ──────────────────────────────────────────────────────────
FOLLOWUP_WINDOW : float = 10.0   # seconds to wait for speech START after a response

# ── TTS settings ──────────────────────────────────────────────────────────────
TTS_MODEL = "tts-1"
TTS_VOICE = "alloy"

# ── Restart flag ──────────────────────────────────────────────────────────────
_restart_flag = threading.Event()


# ─────────────────────────────────────────────────────────────────────────────
#  KEYBOARD MONITOR
# ─────────────────────────────────────────────────────────────────────────────

def _keyboard_monitor(loop: asyncio.AbstractEventLoop) -> None:
    """
    Minimal termios: disables ICANON+ECHO but keeps OPOST so \n → \r\n works.
    Ctrl+R = restart, Ctrl+C = quit.
    """
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        new = termios.tcgetattr(fd)
        new[3] &= ~(termios.ICANON | termios.ECHO)
        new[6][termios.VMIN]  = 1
        new[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, new)
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                ch = os.read(fd, 1)
                if ch == b'\x12':
                    print("\n🔄  Ctrl+R — restarting pipeline…", flush=True)
                    _restart_flag.set()
                    loop.call_soon_threadsafe(lambda: None)
                elif ch == b'\x03':
                    os.kill(os.getpid(), signal.SIGINT)
    except Exception:
        pass
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  BEEP  (aplay — no PyAudio)
# ─────────────────────────────────────────────────────────────────────────────

def play_beep(frequency: int = 880, duration: float = 0.2, volume: float = 0.5) -> None:
    """
    Builds a sine-wave WAV in memory and pipes it to aplay.
    No PyAudio involved — avoids the Pi segfault entirely.
    """
    rate      = 44100
    n_samples = int(rate * duration)
    fade      = int(rate * 0.005)

    buf = array.array('h')
    for i in range(n_samples):
        s = math.sin(2 * math.pi * frequency * i / rate)
        if i < fade:
            s *= i / fade
        elif i > n_samples - fade:
            s *= (n_samples - i) / fade
        buf.append(int(s * volume * 32767))

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(buf.tobytes())

    try:
        subprocess.run(
            ["aplay", "-q", "-D", APLAY_DEVICE, "-"],
            input=wav_io.getvalue(),
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  WAKE WORD
# ─────────────────────────────────────────────────────────────────────────────

def wait_for_wake_word(porcupine: pvporcupine.Porcupine) -> bool:
    recorder = PvRecorder(
        device_index=PVRECORDER_DEVICE_INDEX,
        frame_length=porcupine.frame_length,
    )
    recorder.start()
    print("\n" + "─" * 55)
    print("🔇  Listening for 'Hey Nandi'…  (Ctrl+R to restart, Ctrl+C to quit)")
    print("─" * 55)
    try:
        while True:
            if _restart_flag.is_set():
                return False
            pcm = recorder.read()
            if porcupine.process(pcm) >= 0:
                print("✅  Wake word detected!")
                return True
    finally:
        recorder.stop()
        recorder.delete()


# ─────────────────────────────────────────────────────────────────────────────
#  STT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class _NamedBytesIO(io.BytesIO):
    def __init__(self, data: bytes, name: str = "audio.wav") -> None:
        super().__init__(data)
        self.name = name


def _frames_to_wav(frames: list) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(STT_SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
#  STT — blocking, arecord-based (no PyAudio)
# ─────────────────────────────────────────────────────────────────────────────

def _blocking_capture_and_transcribe(
    sync_openai: OpenAI,
    start_timeout: Optional[float] = None,
) -> str:
    """
    Spawns arecord as a subprocess and reads raw PCM frames from its stdout.
    WebRTC VAD decides when speech starts/ends.
    Sends final audio to OpenAI Whisper for transcription.

    No PyAudio used — avoids the segfault on Python 3.11 / Pi.

    start_timeout:
        Follow-up mode. If user has NOT started speaking within this many
        seconds → return "". Once any speech is detected this is ignored and
        full MAX_SEGMENT_SEC applies — speaking at second 9 of 10s is fine.
    """
    print("\n🎙️  Listening for your command…")

    # arecord: raw signed 16-bit little-endian, mono, 16 kHz
    arecord_cmd = [
        "arecord",
        "-D", ARECORD_DEVICE,
        "-f", "S16_LE",
        "-r", str(STT_SAMPLE_RATE),
        "-c", "1",
        "-t", "raw",
        "-q",          # quiet — suppress progress messages
        "-",           # write to stdout
    ]

    proc = subprocess.Popen(
        arecord_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    vad         = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    pre_roll    = collections.deque(maxlen=PRE_ROLL_FRAMES)
    speech_buf  : list  = []
    in_speech   : bool  = False
    speech_cnt  : int   = 0
    silence_cnt : int   = 0
    start_time  : float = time.monotonic()
    speech_start: float = 0.0

    try:
        while True:
            if _restart_flag.is_set():
                print("\n🔄  STT interrupted by restart.")
                return ""

            # Read exactly one VAD frame from arecord stdout
            raw = b""
            while len(raw) < VAD_FRAME_BYTES:
                chunk = proc.stdout.read(VAD_FRAME_BYTES - len(raw))
                if not chunk:
                    # arecord closed unexpectedly
                    print("\n⚠️  arecord closed unexpectedly.")
                    return ""
                raw += chunk

            is_speech = vad.is_speech(raw, STT_SAMPLE_RATE)
            pre_roll.append(raw)
            now = time.monotonic()

            if not in_speech:
                # Follow-up idle timeout — fires only before speech starts
                if start_timeout and (now - start_time) >= start_timeout:
                    print("\n😴  No speech in follow-up window — going to sleep.")
                    return ""

                if is_speech:
                    speech_cnt += 1
                    if speech_cnt >= SPEECH_TRIGGER_FRAMES:
                        in_speech    = True
                        silence_cnt  = 0
                        speech_start = now
                        speech_buf   = list(pre_roll)   # include pre-roll
                        print("  [speaking…]", end="\r", flush=True)
                else:
                    speech_cnt = 0

            else:
                speech_buf.append(raw)

                if not is_speech:
                    silence_cnt += 1
                    if silence_cnt >= SILENCE_TRIGGER_FRAMES:
                        print("\n🤫  Silence detected — done recording.")
                        break
                else:
                    silence_cnt = 0

                if now - speech_start >= MAX_SEGMENT_SEC:
                    print("\n⏱️  Max recording time reached.")
                    break

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    if len(speech_buf) < MIN_SPEECH_FRAMES:
        print("⚠️  Audio too short — discarding.")
        return ""

    try:
        wav_bytes  = _frames_to_wav(speech_buf)
        audio_file = _NamedBytesIO(wav_bytes)
        kwargs: dict = dict(model="whisper-1", file=audio_file, response_format="text")
        if WHISPER_LANGUAGE:
            kwargs["language"] = WHISPER_LANGUAGE
        text   = sync_openai.audio.transcriptions.create(**kwargs)
        result = text.strip() if isinstance(text, str) else ""
        if result:
            print(f"\r✅  {result}   ")
        return result
    except Exception as e:
        print(f"\n⚠️  Whisper error: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
#  STT — async wrapper
# ─────────────────────────────────────────────────────────────────────────────

async def capture_and_transcribe(
    sync_openai: OpenAI,
    start_timeout: Optional[float] = None,
) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _blocking_capture_and_transcribe, sync_openai, start_timeout
    )


# ─────────────────────────────────────────────────────────────────────────────
#  LLM
# ─────────────────────────────────────────────────────────────────────────────

async def get_llm_response(query: str) -> str:
    print(f"\n📝  You said: \"{query}\"")
    print("🤖  Nandi: ", end="", flush=True)

    for attempt in range(3):
        full_response: list[str] = []
        try:
            async with websockets.connect(LLM_WS_URI) as ws:
                await ws.send(json.dumps({"query": query, "session_id": SESSION_ID}))

                async for raw in ws:
                    if _restart_flag.is_set():
                        print("\n🔄  LLM interrupted by restart.")
                        return ""
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        print(raw, end="", flush=True)
                        full_response.append(raw)
                        continue

                    msg_type = data.get("type")
                    if msg_type == "stream":
                        token = data.get("token", "")
                        print(token, end="", flush=True)
                        full_response.append(token)
                    elif msg_type == "end":
                        print()
                        return "".join(full_response).strip()
                    elif data.get("error"):
                        print(f"\n❌  LLM error: {data['error']}")
                        return ""

                print()
                return "".join(full_response).strip()

        except (websockets.ConnectionClosed, ConnectionRefusedError, OSError) as e:
            print(f"\n⚠️  LLM connection failed (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                await asyncio.sleep(3)
        except Exception as e:
            print(f"\n❌  Unexpected LLM error: {e}")
            break

    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  TTS
# ─────────────────────────────────────────────────────────────────────────────

async def speak(text: str, async_openai: AsyncOpenAI) -> None:
    if not text:
        return
    print("\n🔊  Speaking…")
    try:
        ffplay = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-hide_banner", "-loglevel", "error", "-"],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        async with async_openai.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
        ) as response:
            async for chunk in response.iter_bytes(chunk_size=4096):
                if _restart_flag.is_set():
                    print("\n🔄  TTS interrupted by restart.")
                    ffplay.kill()
                    return
                if ffplay.stdin:
                    ffplay.stdin.write(chunk)
                    ffplay.stdin.flush()

        if ffplay.stdin:
            ffplay.stdin.close()
        ffplay.wait()

    except FileNotFoundError:
        print("⚠️  ffplay not found — run: sudo apt install ffmpeg -y")
    except Exception as e:
        print(f"❌  TTS failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

async def assistant_loop():
    if not ACCESS_KEY:
        raise ValueError("PICOVOICE_ACCESS_KEY not set in .env")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in .env")

    porcupine    = pvporcupine.create(access_key=ACCESS_KEY, keyword_paths=[MODEL_PATH])
    async_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)
    sync_openai  = OpenAI(api_key=OPENAI_API_KEY)
    loop         = asyncio.get_event_loop()

    threading.Thread(
        target=_keyboard_monitor, args=(loop,), daemon=True, name="kbd-monitor"
    ).start()

    print("=" * 55)
    print("  🎤  Nandi Voice Assistant — Raspberry Pi")
    print("  💡  Press Ctrl+R anytime to restart the pipeline")
    print("=" * 55)

    try:
        while True:
            _restart_flag.clear()

            # 1. Wake word
            detected = await loop.run_in_executor(None, wait_for_wake_word, porcupine)
            if not detected:
                continue

            # 2. Beep → signals ready to listen
            await loop.run_in_executor(None, play_beep)
            if _restart_flag.is_set():
                continue

            # 3. Conversation loop (no wake word for follow-ups)
            is_followup = False
            while True:
                if _restart_flag.is_set():
                    break

                if is_followup:
                    print(f"\n💬  Follow-up window ({FOLLOWUP_WINDOW:.0f}s)… "
                          "speak or stay silent to sleep.")
                    await loop.run_in_executor(
                        None, lambda: play_beep(frequency=660, duration=0.12)
                    )
                    if _restart_flag.is_set():
                        break
                    transcription = await capture_and_transcribe(
                        sync_openai, start_timeout=FOLLOWUP_WINDOW
                    )
                else:
                    transcription = await capture_and_transcribe(sync_openai)

                if _restart_flag.is_set():
                    break

                if not transcription:
                    print("😴  Nothing heard — going back to sleep.")
                    break

                response = await get_llm_response(transcription)
                if _restart_flag.is_set():
                    break
                if not response:
                    print("⚠️  Empty LLM response — going back to sleep.")
                    break

                await speak(response, async_openai)
                is_followup = True

    except KeyboardInterrupt:
        print("\n\n👋  Shutting down Nandi…")
    finally:
        porcupine.delete()
        print("✅  Cleanup complete.")


if __name__ == "__main__":
    try:
        asyncio.run(assistant_loop())
    except KeyboardInterrupt:
        pass

"""
Real-Time Speech-to-Text â€” WebRTC VAD + OpenAI Whisper
=======================================================
Drop-in replacement for the Silero-based version.
Uses Google WebRTC VAD â€” the same engine inside Chrome & Android.
Compiles natively on ARM: no AVX/SSE4 needed. Works on ALL Raspberry
Pi models and Windows without modification.

Audio never touches disk: PCM -> in-memory WAV -> Whisper API.

INSTALL
-------
Windows:
    pip install openai pyaudio numpy webrtcvad

Raspberry Pi:
    sudo apt install portaudio19-dev -y
    pip install openai pyaudio numpy webrtcvad
    # if webrtcvad fails to compile:
    pip install webrtcvad-wheels

USAGE
-----
    export OPENAI_API_KEY=sk-...   # Linux / macOS / Pi
    set    OPENAI_API_KEY=sk-...   # Windows CMD
    python realtime_stt_vad.py
    Ctrl+C to stop.
"""

from __future__ import annotations

import collections
import io
import os
import queue
import threading
import time
import wave
from typing import Optional

import pyaudio
import webrtcvad
from openai import OpenAI

# =============================================================================
#  CONFIGURATION  â€” tune these to match your environment
# =============================================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")

# -- Audio --------------------------------------------------------------------
SAMPLE_RATE       : int   = 16_000   # Hz  (Whisper & WebRTC VAD both want 16 k)
CHANNELS          : int   = 1        # Mono only

# WebRTC VAD requires frames of exactly 10 / 20 / 30 ms
VAD_FRAME_MS      : int   = 30       # 30 ms -> best balance of latency & accuracy
VAD_FRAME_SAMPLES : int   = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)   # 480
VAD_FRAME_BYTES   : int   = VAD_FRAME_SAMPLES * 2                     # 960

# -- WebRTC VAD aggressiveness ------------------------------------------------
#   0 = least aggressive (quietest rooms, fewest false-negatives)
#   1 = gentle filtering
#   2 = moderate  <- good default for Raspberry Pi with USB mic
#   3 = most aggressive (loud fans, traffic, open offices)
VAD_AGGRESSIVENESS    : int   = 3

# -- Speech / silence timing --------------------------------------------------
SPEECH_TRIGGER_FRAMES : int   = 6    # consecutive voiced frames before onset  (~90 ms)
SILENCE_TRIGGER_FRAMES: int   = 20   # consecutive silent frames to end segment (~600 ms)
MAX_SEGMENT_SEC       : float = 25.0 # force-flush if speech goes on too long
MIN_SPEECH_FRAMES     : int   = 15   # drop clips shorter than this            (~240 ms)

# -- Pre-roll -----------------------------------------------------------------
# Frames captured BEFORE speech onset are prepended to the segment so the
# very first phoneme is never clipped.
PRE_ROLL_FRAMES       : int   = 10   # 10 x 30 ms = 300 ms

# -- Whisper ------------------------------------------------------------------
WHISPER_LANGUAGE      : Optional[str] = "en"   # None = auto-detect


# =============================================================================
#  HELPERS
# =============================================================================

class NamedBytesIO(io.BytesIO):
    """BytesIO with a .name attribute so the OpenAI SDK can infer MIME type."""
    def __init__(self, data: bytes, name: str = "audio.wav") -> None:
        super().__init__(data)
        self.name = name


def frames_to_wav_bytes(frames: list) -> bytes:
    """Convert raw 16-bit PCM frames to a WAV byte-string in RAM (no disk)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


# =============================================================================
#  TRANSCRIPTION WORKER  (background thread)
# =============================================================================

def transcription_worker(
    job_queue  : queue.Queue,
    stop_event : threading.Event,
    client     : OpenAI,
) -> None:
    """Pull WAV blobs from the queue and send them to Whisper."""
    while not stop_event.is_set() or not job_queue.empty():
        try:
            wav_bytes: bytes = job_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            audio_file = NamedBytesIO(wav_bytes, "audio.wav")
            kwargs: dict = dict(
                model="whisper-1",
                file=audio_file,
                response_format="text",
            )
            if WHISPER_LANGUAGE:
                kwargs["language"] = WHISPER_LANGUAGE

            text = client.audio.transcriptions.create(**kwargs)
            if isinstance(text, str):
                text = text.strip()

            if text:
                print(f"\n  TRANSCRIPT: {text}")
                print("  " + "-" * 60)

        except Exception as exc:
            print(f"\n  [Whisper error] {exc}")

        finally:
            job_queue.task_done()


# =============================================================================
#  MAIN
# =============================================================================

def run() -> None:

    # Initialise WebRTC VAD
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    # OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    # PyAudio stream
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=VAD_FRAME_SAMPLES,
    )

    # Threading
    job_queue  : queue.Queue     = queue.Queue()
    stop_event : threading.Event = threading.Event()
    worker = threading.Thread(
        target=transcription_worker,
        args=(job_queue, stop_event, client),
        daemon=True,
    )
    worker.start()

    # State machine variables
    pre_roll     = collections.deque(maxlen=PRE_ROLL_FRAMES)
    speech_buf   : list  = []
    in_speech    : bool  = False
    speech_cnt   : int   = 0
    silence_cnt  : int   = 0
    speech_start : float = 0.0

    def flush() -> None:
        nonlocal speech_buf, in_speech, speech_cnt, silence_cnt
        if len(speech_buf) >= MIN_SPEECH_FRAMES:
            job_queue.put(frames_to_wav_bytes(speech_buf))
            print("  [-> Whisper (in-memory, no file)]       ", end="\r")
        speech_buf  = []
        in_speech   = False
        speech_cnt  = 0
        silence_cnt = 0

    print("\n" + "=" * 64)
    print("  Real-Time STT  |  WebRTC VAD  +  OpenAI Whisper")
    print("=" * 64)
    print(f"  VAD aggressiveness : {VAD_AGGRESSIVENESS}   (0=gentle  3=strict)")
    print(f"  Frame              : {VAD_FRAME_MS} ms  |  {SAMPLE_RATE} Hz mono")
    print(f"  End-of-speech gap  : {SILENCE_TRIGGER_FRAMES} frames = "
          f"{SILENCE_TRIGGER_FRAMES * VAD_FRAME_MS} ms silence")
    print(f"  Pre-roll           : {PRE_ROLL_FRAMES} frames = "
          f"{PRE_ROLL_FRAMES * VAD_FRAME_MS} ms before onset")
    print("  No temp files -- audio encoded in RAM.")
    print("  Works on Raspberry Pi (all models) + Windows.")
    print("  Speak now...  Ctrl+C to stop.")
    print("-" * 64 + "\n")

    try:
        while True:
            raw = stream.read(VAD_FRAME_SAMPLES, exception_on_overflow=False)

            # Pad to exact length on short reads (can happen on Pi)
            if len(raw) < VAD_FRAME_BYTES:
                raw = raw + b"\x00" * (VAD_FRAME_BYTES - len(raw))

            is_speech = vad.is_speech(raw, SAMPLE_RATE)
            pre_roll.append(raw)

            if not in_speech:
                # Waiting for speech onset
                if is_speech:
                    speech_cnt += 1
                    if speech_cnt >= SPEECH_TRIGGER_FRAMES:
                        in_speech    = True
                        silence_cnt  = 0
                        speech_start = time.monotonic()
                        speech_buf   = list(pre_roll)   # prepend pre-roll
                        print("  [speaking...]                          ", end="\r")
                else:
                    speech_cnt = 0

            else:
                # Inside a speech segment
                speech_buf.append(raw)

                if not is_speech:
                    silence_cnt += 1
                    if silence_cnt >= SILENCE_TRIGGER_FRAMES:
                        flush()
                else:
                    silence_cnt = 0

                # Safety flush on max duration
                if time.monotonic() - speech_start >= MAX_SEGMENT_SEC:
                    print("  [max duration -> flush]                ", end="\r")
                    flush()

    except KeyboardInterrupt:
        print("\n\n  Stopping...\n")
        if in_speech and speech_buf:
            flush()

    finally:
        stop_event.set()
        job_queue.join()
        stream.stop_stream()
        stream.close()
        pa.terminate()
        print("  Done. Goodbye!\n")


# =============================================================================

if __name__ == "__main__":
    if OPENAI_API_KEY == "sk-YOUR_API_KEY_HERE":
        print(
            "\n  No API key found.\n"
            "  Set env var OPENAI_API_KEY, or edit the constant at\n"
            "  the top of this file.\n"
        )
    else:
        run()
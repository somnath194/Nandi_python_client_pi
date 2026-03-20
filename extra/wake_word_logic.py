from dotenv import load_dotenv
import pvporcupine
from pvrecorder import PvRecorder
from pathlib import Path
import os

access_key_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=access_key_path, override=True)
ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")

MODEL_PATH = "/home/somi/Documents/nandi_python_client/data/hay-nandi_en_raspberry-pi_v4_0_0.ppn"

porcupine = pvporcupine.create(
    access_key=ACCESS_KEY, # type: ignore
    keyword_paths=[MODEL_PATH]
)
recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)

print("Listening for 'Hey Nandi'...")

try:
    recorder.start()
    while True:
        pcm = recorder.read()
        if porcupine.process(pcm) >= 0:
            print("Wake word 'Hey Nandi' detected!")
except KeyboardInterrupt:
    recorder.stop()
finally:
    recorder.delete()
    porcupine.delete()


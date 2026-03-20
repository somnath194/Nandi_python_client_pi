import os
import base64
import subprocess
import asyncio
from datetime import datetime
from pathlib import Path
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
dotenv_path = Path(__file__).resolve().parents[1] /'.env' # adjust if needed
# dotenv_path = ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)

api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)  # type: ignore


class TTSPlayer:
    def __init__(self, model: str = "tts-1", voice: str = "alloy"):
        self.model = model
        self.voice = voice

    async def play(self, text: str):
        """
        Stream TTS audio and play it in real-time with ffplay.
        """
        try:
            # Start ffplay process
            ffplay = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-hide_banner", "-loglevel", "error", "-"],
                stdin=subprocess.PIPE,
            )

            async with client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=self.voice,
                input=text,
            ) as response:
                async for chunk in response.iter_bytes(chunk_size=4096):
                    if ffplay.stdin:
                        ffplay.stdin.write(chunk)
                        ffplay.stdin.flush()

            if ffplay.stdin:
                ffplay.stdin.close()
            ffplay.wait()

            print(f"✅ Finished playing TTS at {datetime.now().isoformat()}")

        except Exception as e:
            print(f"❌ TTS failed: {str(e)}")


# Example usage
async def main():
    tts = TTSPlayer()
    while True:
        text = input("> ").strip()
        if not text:
            break
        await tts.play(text)


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import websockets
import json
import os
import time

WS_URI = "ws://localhost:8000/ws/chat_stream"

EVENT_FILE = "/home/somi/Documents/nandi_python_client/data/clap_event.txt"

RETRY_DELAY = 5

Query = "turn on my pc and turn on my bedroom fan"
# Query = "hello, nandi"



async def clap_watcher(ws):
    """Watch clap_event.txt and send query when it changes"""
    print(f"Watching clap event file: {EVENT_FILE}")

    # Get current timestamp when script starts
    last_mtime = None

    if os.path.exists(EVENT_FILE):
        last_mtime = os.path.getmtime(EVENT_FILE)

    while True:
        try:
            if os.path.exists(EVENT_FILE):
                mtime = os.path.getmtime(EVENT_FILE)

                # Only trigger when file changes AFTER script start
                if last_mtime is not None and mtime != last_mtime:
                    last_mtime = mtime

                    print("Double clap detected ? activating work mode")

                    payload = {
                        "query": Query,
                        "session_id": "test-session"
                    }

                    await ws.send(json.dumps(payload))

                else:
                    last_mtime = mtime

        except Exception as e:
            print("?? Clap watcher error:", e)

        await asyncio.sleep(0.5)


async def receiver(ws):
    """Receive streamed responses from AI assistant"""
    try:
        async for message in ws:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                print("\n?? Raw:", message)
                continue

            msg_type = data.get("type")

            if msg_type == "stream":
                token = data.get("token", "")
                print(token, end="", flush=True)

            elif msg_type == "end":
                print("\n? Response complete")

            else:
                print("\n??", data)

    except websockets.ConnectionClosed as e:
        print("Connection closed:", e)
        raise


async def connect_forever():
    """Keep websocket connected forever"""
    while True:
        try:
            print("Connecting to AI assistant...")

            async with websockets.connect(WS_URI) as ws:

                print("Connected to AI assistant")
                print("Waiting for clap events...\n")

                await asyncio.gather(
                    clap_watcher(ws),
                    receiver(ws)
                )

        except Exception as e:
            print("Connection lost:", e)
            print(f"Reconnecting in {RETRY_DELAY} seconds...\n")
            await asyncio.sleep(RETRY_DELAY)


if __name__ == "__main__":
    asyncio.run(connect_forever())
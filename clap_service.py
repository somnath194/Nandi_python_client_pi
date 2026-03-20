import subprocess
import time

CMD = ["csound", "/home/somi/Documents/nandi_python_client/clap-detection/clap.csd"]

while True:
    try:
        print("Starting Csound clap detector...")
        subprocess.call(CMD)
    except Exception as e:
        print("Csound crashed:", e)

    print("Restarting in 3 seconds...")
    time.sleep(3)

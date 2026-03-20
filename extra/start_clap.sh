#!/bin/bash

cd /home/somi/Documents/nandi_python_client/clap-detection-master

sleep 10

# Terminal 1 ? Clap detector
lxterminal --title="Clap Detector" -e python3 /home/somi/Documents/nandi_python_client/clap_service.py &
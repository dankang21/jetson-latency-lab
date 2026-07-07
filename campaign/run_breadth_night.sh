#!/bin/bash
# E-B breadth night run: vision breadth then Qwen-3B, sequential.
cd /home/dk/mobile/jetson-latency-lab
echo "[$(date)] breadth_deploy start"
bash campaign/breadth_deploy.sh
echo "[$(date)] qwen3b_deploy start"
bash campaign/qwen3b_deploy.sh
echo "[$(date)] NIGHT RUN ALL DONE"

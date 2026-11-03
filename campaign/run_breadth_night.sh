#!/bin/bash
# E-B breadth night run: vision breadth then Qwen-3B, sequential.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
echo "[$(date)] breadth_deploy start"
bash campaign/breadth_deploy.sh
echo "[$(date)] qwen3b_deploy start"
bash campaign/qwen3b_deploy.sh
echo "[$(date)] NIGHT RUN ALL DONE"

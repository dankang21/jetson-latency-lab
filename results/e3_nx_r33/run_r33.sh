#!/bin/bash
# post-hoc r33 sensitivity runs (E3_protocol.md §8, 2026-10-23). Sequential; each step logs to run.log.
cd /home/dk/mobile/jetson-latency-lab
C="results/refit33/nx/cells_mobilenet/nx_est33_emc2133_gpu306_mobilenet.json results/refit33/nx/cells_mobilenet/nx_est33_emc2133_gpu1173_mobilenet.json"
L=results/e3_nx_r33/run.log
echo "start $(date -Is)" >> $L
./campaign/admit_refsweep.sh --primary mobilenet --co vit --co-hz 10 --D 12.0 --cond C0r33 --outroot results/e3_nx_r33 >> $L 2>&1; echo "refsweep C0r33 rc=$? $(date -Is)" >> $L
./campaign/admit_loop.sh --primary mobilenet --co vit --co-hz 10 --D 12.0 --trials 10 --cond C0r33 --outroot results/e3_nx_r33 --cells "$C" >> $L 2>&1; echo "loop C0r33 rc=$? $(date -Is)" >> $L
./campaign/admit_refsweep.sh --primary mobilenet --co resnet50 --co-hz 5 --D 12.0 --cond N3r33 --outroot results/e3_nx_r33 >> $L 2>&1; echo "refsweep N3r33 rc=$? $(date -Is)" >> $L
./campaign/admit_loop.sh --primary mobilenet --co resnet50 --co-hz 5 --D 12.0 --trials 20 --cond N3r33 --outroot results/e3_nx_r33 --cells "$C" >> $L 2>&1; echo "loop N3r33 rc=$? $(date -Is)" >> $L
echo "ALL DONE $(date -Is)" >> $L

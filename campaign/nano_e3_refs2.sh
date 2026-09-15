#!/bin/bash
cd /home/dk/mobile/jetson-latency-lab; L=./campaign/admit_loop.sh; S=./campaign/admit_refsweep.sh
$S --primary vit --co yolov8s --co-hz 5  --D 37.5 --cond A5r
$S --primary vit --co yolov8s --co-hz 10 --D 37.5 --cond A6r
$L --primary vit --co yolov8s --co-hz 7  --D 37.5 --trials 10 --cond A7
$S --primary vit --co yolov8s --co-hz 7  --D 37.5 --cond A7
echo "NANO_REFS2 DONE $(date -Is)"

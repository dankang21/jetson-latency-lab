#!/bin/bash
# E3 Nano conditions A1-A6 (protocol §3) then reference sweeps; resumable.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
L=./campaign/admit_loop.sh; S=./campaign/admit_refsweep.sh
$L --primary mobilenet --co yolov8s  --co-hz 5  --D 13.0 --trials 10 --cond A1
$L --primary mobilenet --co yolov8s  --co-hz 10 --D 13.0 --trials 10 --cond A2
$L --primary mobilenet --co resnet50 --co-hz 5  --D 13.0 --trials 10 --cond A3
$L --primary mobilenet --co resnet50 --co-hz 10 --D 13.0 --trials 10 --cond A4
$L --primary vit       --co yolov8s  --co-hz 5  --D 37.5 --trials 10 --cond A5
$L --primary vit       --co yolov8s  --co-hz 10 --D 37.5 --trials 10 --cond A6
$S --primary mobilenet --co yolov8s  --co-hz 5  --D 13.0 --cond A1
$S --primary mobilenet --co yolov8s  --co-hz 10 --D 13.0 --cond A2
$S --primary mobilenet --co resnet50 --co-hz 5  --D 13.0 --cond A3
$S --primary mobilenet --co resnet50 --co-hz 10 --D 13.0 --cond A4
$S --primary vit       --co yolov8s  --co-hz 5  --D 37.5 --cond A5
$S --primary vit       --co yolov8s  --co-hz 10 --D 37.5 --cond A6
echo "NANO_E3 DONE $(date -Is)"

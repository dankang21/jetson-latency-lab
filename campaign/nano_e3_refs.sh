#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1; S=./campaign/admit_refsweep.sh
$S --primary mobilenet --co yolov8s  --co-hz 10 --D 13.0 --cond A2
$S --primary mobilenet --co resnet50 --co-hz 5  --D 13.0 --cond A3
$S --primary mobilenet --co resnet50 --co-hz 10 --D 13.0 --cond A4
$S --primary vit       --co yolov8s  --co-hz 5  --D 37.5 --cond A5
$S --primary vit       --co yolov8s  --co-hz 10 --D 37.5 --cond A6
echo "NANO_REFS DONE $(date -Is)"

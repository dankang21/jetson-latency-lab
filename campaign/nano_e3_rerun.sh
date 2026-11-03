#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1; L=./campaign/admit_loop.sh; S=./campaign/admit_refsweep.sh
$L --primary vit --co yolov8s --co-hz 5  --D 37.5 --trials 10 --cond A5r
$L --primary vit --co yolov8s --co-hz 10 --D 37.5 --trials 10 --cond A6r
$S --primary vit --co yolov8s --co-hz 5  --D 37.5 --cond A5r
$S --primary vit --co yolov8s --co-hz 10 --D 37.5 --cond A6r
echo "NANO_RERUN DONE $(date -Is)"

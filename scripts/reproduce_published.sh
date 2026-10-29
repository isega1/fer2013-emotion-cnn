#!/usr/bin/env bash
# Recompute the numbers quoted in the README from the shipped checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

for epoch in 5 10; do
  $PY -m src.evaluate --checkpoint "checkpoints/model_epoch_${epoch}.pth" \
      --out "results/metrics_epoch${epoch}.json" \
      --figure "results/figures/confusion_matrix_epoch${epoch}.png"
done
$PY -m src.export_onnx --checkpoint checkpoints/model_epoch_10.pth
$PY -m src.benchmark --onnx checkpoints/emotioncnn.onnx

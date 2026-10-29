#!/usr/bin/env bash
# Recompute every row of the README results table from the published
# checkpoints. The weights are release assets, not tracked files:
#
#   gh release download v1.0-checkpoints -D checkpoints
#
# Any asset that is missing is skipped with a message instead of aborting, so a
# partial download still verifies the rows it can.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

need() {
  if [ -f "$1" ]; then
    return 0
  fi
  echo "skip: $1 not found — run: gh release download v1.0-checkpoints -D checkpoints"
  return 1
}

# legacy split, 5 and 10 epochs (59.43%, 63.39%)
for epoch in 5 10; do
  if need "checkpoints/model_epoch_${epoch}.pth"; then
    "$PY" -m src.evaluate --checkpoint "checkpoints/model_epoch_${epoch}.pth" \
        --split data/fer2013/test.csv \
        --out "results/metrics_epoch${epoch}.json" \
        --figure "results/figures/confusion_matrix_epoch${epoch}.png"
  fi
done

# headline rows: 63.81% (official) and 63.95% (legacy, 40 epochs)
if need checkpoints/cnn40-official_best.pth; then
  "$PY" -m src.evaluate --checkpoint checkpoints/cnn40-official_best.pth \
      --split data/fer2013_official/test.csv \
      --out results/metrics_official.json \
      --figure results/figures/cm_official.png
fi

if need checkpoints/cnn40-legacy_best.pth; then
  "$PY" -m src.evaluate --checkpoint checkpoints/cnn40-legacy_best.pth \
      --split data/fer2013/test.csv --out results/metrics_legacy40.json
fi

# ONNX parity + CPU latency (63.39% row's deployment numbers)
if need checkpoints/model_epoch_10.pth; then
  "$PY" -m src.export_onnx --checkpoint checkpoints/model_epoch_10.pth
  "$PY" -m src.benchmark --onnx checkpoints/emotioncnn.onnx
fi

#!/usr/bin/env bash
# Full pipeline: split -> train -> evaluate -> export -> benchmark.
# Override with env vars, e.g. EPOCHS=60 MODEL=cnn-small ./scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
MODEL=${MODEL:-emotioncnn}
EPOCHS=${EPOCHS:-40}
TAG=${TAG:-$MODEL}

$PY -m src.split --mode "${SPLIT_MODE:-legacy}"
$PY -m src.train --model "$MODEL" --tag "$TAG" --epochs "$EPOCHS"
$PY -m src.evaluate --checkpoint "checkpoints/${TAG}_best.pth" \
    --figure "results/figures/confusion_matrix_${TAG}.png" \
    --out "results/metrics_${TAG}.json"
$PY -m src.export_onnx --checkpoint "checkpoints/${TAG}_best.pth" \
    --out "checkpoints/${TAG}.onnx"
$PY -m src.benchmark --onnx "checkpoints/${TAG}.onnx" \
    --out "results/benchmark_${TAG}.json"

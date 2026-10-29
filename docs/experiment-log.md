# Experiment log

One row per run. `scripts/reproduce_published.sh` regenerates the first two
rows from the shipped checkpoints; `scripts/run_all.sh` produces new ones.

| date | run | config | test acc | macro-F1 | notes |
| --- | --- | --- | --- | --- | --- |
| 2025-01 | epoch 5 | Adam 1e-3, batch 32, CPU | 59.43% | 0.5354 | original coursework checkpoint |
| 2025-01 | epoch 10 | Adam 1e-3, batch 32, CPU | 63.39% | 0.6046 | original coursework checkpoint, still improving |
| 2026-10-29 | reproduction | same checkpoints, `src/evaluate.py` | 63.39% | 0.6046 | matches the 2025 numbers exactly; `src/split.py --mode legacy` reproduces the shipped test rows (same 3,589 rows, same order, LF vs CRLF only) |
| 2026-10-29 | ONNX export | opset 17, batch dynamic | 63.39% | 0.6046 | parity with PyTorch: max |Δlogit| = 5.2e-06 |
| 2026-10-29 | benchmark | 1 thread, 50 runs | — | — | 1.07 ms vs 1.71 ms per image (1.60×) |

## Planned

- 40 epochs with cosine decay on the legacy split (does the 4-point gain continue?).
- `--mode official` evaluation on the untouched PublicTest/PrivateTest rows: the
  number that can be compared with published work.
- Class weighting or focal loss, target: disgust and sad recall.
- 6-model comparison table (CNN light/small vs MLP small/medium/large).

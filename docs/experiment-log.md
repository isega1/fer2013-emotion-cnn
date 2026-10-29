# Experiment log

One row per run. `scripts/reproduce_published.sh` regenerates the first four
rows from the shipped checkpoints; `scripts/run_all.sh` produces new ones.

All rows dated 2026-10-29 are reproduced from scratch on an RTX 5070 Ti
(torch 2.11.0+cu128, CUDA 12.8, Windows 11) from a fresh download of the
FER2013 CSV; the two 2025-01 rows come from the original coursework run.

| date | run | config | test acc | macro-F1 | notes |
| --- | --- | --- | --- | --- | --- |
| 2025-01 | epoch 5 | Adam 1e-3, batch 32, CPU | 59.43% | 0.5354 | original coursework checkpoint |
| 2025-01 | epoch 10 | Adam 1e-3, batch 32, CPU | 63.39% | 0.6046 | original coursework checkpoint, still improving |
| 2026-10-29 | reproduction | same checkpoints, `src/evaluate.py` | 63.39% | 0.6046 | matches the 2025 numbers exactly; `src/split.py --mode legacy` reproduces the shipped test rows (same 3,589 rows, same order, LF vs CRLF only) |
| 2026-10-29 | ONNX export | opset 17, batch dynamic | 63.39% | 0.6046 | parity with PyTorch: max abs(Δlogit) = 5.2e-06 |
| 2026-10-29 | benchmark | 1 thread, 50 runs | — | — | 1.07 ms vs 1.71 ms per image (1.60×) |
| 2026-10-29 | CPU run, aborted | Adam 1e-3, batch 128, 8 threads, legacy | — | — | stopped at epoch 4 (val 59.10%) after 81 s/epoch; machine moved to a CUDA box, see `results/train_emotioncnn40.log` |
| 2026-10-29 | 40 epochs, official | emotioncnn, official split, Adam 1e-3, batch 256, cosine, 40 ep, CUDA | **63.81%** | **0.6295** | first number that may be compared with published work (7,178 untouched PublicTest+PrivateTest rows); best val 62.97% @ epoch 30; `results/metrics_official.json` |
| 2026-10-29 | 40 epochs, legacy | emotioncnn, legacy split, Adam 1e-3, batch 256, cosine, 40 ep, CUDA | 63.95% | 0.6310 | same protocol as the 63.39% row: +0.56 pp for 4× the epochs; 0.14 pp above the official split; `results/metrics_legacy40.json` |
| 2026-10-29 | 6-model: cnn-light | official split, 20 ep, batch 256, cosine | 60.43% | 0.5937 | 939,351 params, 0.62 ms/image |
| 2026-10-29 | 6-model: cnn-small | official split, 20 ep, batch 256, cosine | 60.57% | 0.5951 | 1,274,823 params, 0.90 ms/image |
| 2026-10-29 | 6-model: emotioncnn | official split, 20 ep, batch 256, cosine | 63.00% | 0.6239 | 10,927,751 params, 3.74 ms/image (the "deep" slot; no `cnn-deep` model exists in `src/model.py`) |
| 2026-10-29 | 6-model: mlp-small | official split, 20 ep, batch 256, cosine | 45.96% | 0.4320 | 1,183,751 params, 0.06 ms/image |
| 2026-10-29 | 6-model: mlp-medium | official split, 20 ep, batch 256, cosine | 46.99% | 0.4605 | 3,417,095 params, 0.15 ms/image |
| 2026-10-29 | 6-model: mlp-large | official split, 20 ep, batch 256, cosine | 48.72% | 0.4865 | 8,931,335 params, 0.74 ms/image |

Latency for the 6-model table is `src/latency.py` with the settings the
published 1.71 ms figure used (1 thread, batch 1, 50 runs after 10 warm-up
runs), measured on an AMD Ryzen 7 9700X, not on the Apple silicon machine
that produced the PyTorch-vs-ONNX table — the two sets of millisecond numbers
are not comparable with each other, only within themselves.

## Findings

- **40 epochs do not beat 10 epochs by much.** Legacy goes 63.39% → 63.95%
  (+0.56 pp) and the validation curve in `results/figures/curves_cnn40-legacy.png`
  is flat from roughly epoch 8. The gap between the two splits is only 0.14 pp
  (63.95% legacy vs 63.81% official), so for this model the random-split
  leakage turns out to be a smaller effect than the difference between
  architectures — the methodological objection stands, the size of the
  inflation is simply not dramatic here.
- **The model overfits early.** Train loss reaches 0.02 while validation loss
  climbs from 1.1 to 2.44 (`results/figures/curves_cnn40-official.png`): the
  remaining error is not a duration problem, it is capacity plus
  augmentation plus class imbalance.
- **Parameter count is nearly free, latency is not.** cnn-light (0.94 M
  params) is within 3 pp of emotioncnn (10.9 M) at 6× less CPU time and 11.6×
  fewer parameters; MLPs are 5–60× faster than the CNNs and 12–18 pp worse.

## Planned

- Class weighting or focal loss, target: disgust and sad recall (both are at
  ~0.5 F1 and drag macro-F1 down more than their 1.5%–9% share suggests).
- Horizontal flip / small affine augmentation, evaluated on the official split.
- Publish the two original checkpoints (release asset or Git LFS): without
  them `scripts/reproduce_published.sh` cannot be run by anyone else, which
  undercuts the "every number recomputable" claim.
- If a genuinely deeper CNN is wanted, it has to be written —
  `src/model.py` has six registered models and `cnn-deep` is not one of them.

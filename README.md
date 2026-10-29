# FER2013 Emotion Recognition

A 7-class facial expression classifier on FER2013, rebuilt from a 2025 course
project so that every number in it can be recomputed by someone else.

Original work: *Fundamentals of Artificial Intelligence*, Zhejiang University,
January 2025. This repository adds the parts the course project skipped —
a reproducible evaluation, a proper ONNX export with numerical verification,
and a CPU latency benchmark.

## Results

Evaluation on the 3,589-image held-out split (`data/fer2013/test.csv`),
computed by `src/evaluate.py` from the shipped checkpoints:

| checkpoint | accuracy | macro-F1 | weighted-F1 |
| --- | --- | --- | --- |
| `model_epoch_5.pth` | 59.43% | 0.5354 | 0.5810 |
| `model_epoch_10.pth` | **63.39%** | **0.6046** | 0.6278 |

Majority-class baseline (`happy`) is 24.55%. Chance is 14.3%.

Per class, epoch 10:

| class | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| angry | 0.530 | 0.556 | 0.543 | 498 |
| disgust | 0.667 | 0.462 | 0.545 | 52 |
| fear | 0.525 | 0.439 | 0.478 | 545 |
| happy | 0.803 | 0.862 | 0.831 | 881 |
| sad | 0.537 | 0.420 | 0.471 | 588 |
| surprise | 0.751 | 0.758 | 0.755 | 414 |
| neutral | 0.552 | 0.679 | 0.609 | 611 |

Accuracy is reported alongside macro-F1 because the classes are badly
imbalanced (52 disgust images against 881 happy): a model can raise accuracy
by ignoring the small classes, and macro-F1 will not.

## CPU latency

`src/benchmark.py`, 1 thread, 50 runs after 10 warm-up runs, Apple silicon
(arm64), PyTorch 2.15.0 vs ONNX Runtime 1.31.1:

| batch | PyTorch (median) | ONNX Runtime (median) | speed-up |
| --- | --- | --- | --- |
| 1 | 1.71 ms | 1.07 ms | 1.60× |
| 32 | 60.57 ms | 36.95 ms | 1.64× |

The ONNX export is verified, not assumed: `src/export_onnx.py` runs both
runtimes on the same batch and fails unless the logits agree within 1e-5
(measured difference: 5.2e-06).

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# data (not in this repo): FER2013 CSV from Kaggle, one file
#   data/fer2013/fer2013.csv   (columns: emotion, pixels, Usage)

.venv/bin/python -m src.split --mode legacy        # rebuild the split files
./scripts/reproduce_published.sh                   # evaluate + export + benchmark
```

Full pipeline (retrain from scratch, ~40 epochs, CPU is enough):

```bash
EPOCHS=40 ./scripts/run_all.sh
```

## Repository layout

```
src/dataset.py      CSV -> (N, 48, 48) uint8 -> tensor in [-1, 1]
src/model.py        EmotionCNN (10,929,163 parameters) + lighter variants
src/split.py        legacy random split or the official Usage-based split
src/train.py        training loop with per-epoch validation accuracy
src/evaluate.py     accuracy, macro-F1, per-class metrics, confusion matrix
src/export_onnx.py  ONNX export + numerical parity check
src/benchmark.py    CPU latency, PyTorch vs ONNX Runtime
scripts/            one-command pipelines
results/            metrics JSON and figures produced by the scripts
docs/               experiment log
```

## Known limitations

These are stated up front because they bound what the numbers mean.

1. **The split is a random one, not the official test set.** The 2025 code did
   `train_test_split(df, test_size=0.1, random_state=42)` twice, which ignores
   the `Usage` column FER2013 ships with. FER2013 images are not independent
   (the same people and near-duplicate frames appear across rows), so a random
   split leaks identities between train and test and inflates the score.
   `src/split.py --mode official` provides the honest alternative; the number
   above is the legacy one, kept for comparability with the original report.
2. **10 epochs is not convergence.** Validation accuracy was still rising when
   training stopped (59.4% at epoch 5, 63.4% at epoch 10). `src/train.py`
   defaults to 40 epochs with cosine decay for this reason.
3. **No augmentation, no class weighting.** The class imbalance is partly
   compensated for in reporting (macro-F1) but not in training.
4. **Latency is desktop CPU, not an edge device.** 1.1 ms on a laptop says
   little about a Jetson-class board under an INT8 runtime.

## Data and license

FER2013 comes from the ICML 2013 representation learning challenge
(Kaggle: `deadskull7/fer2013`) and is not redistributed here. Code in this
repository is MIT licensed.

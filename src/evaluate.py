"""Evaluate a checkpoint on a split and write machine readable metrics.

Usage
-----
python -m src.evaluate --checkpoint checkpoints/model_epoch_10.pth \
    --split data/fer2013/test.csv --out results/metrics.json

Everything reported is computed here: accuracy, macro F1, per class
precision/recall/F1/support and the confusion matrix. Nothing is copied
from the training log.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader

from src.dataset import EMOTIONS, FER2013Dataset
from src.model import build_model, count_parameters

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, logits = [], []
    for x, y in loader:
        out = model(x.to(device))
        logits.append(out.cpu())
        labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(logits).numpy()


def summarize(labels: np.ndarray, logits: np.ndarray) -> dict:
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = probs.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=range(len(EMOTIONS)), zero_division=0
    )
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "weighted_f1": float(f1_score(labels, preds, average="weighted")),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(EMOTIONS)
        },
        "confusion_matrix": confusion_matrix(labels, preds, labels=range(len(EMOTIONS))).tolist(),
        "majority_class_baseline": float(np.bincount(labels).max() / len(labels)),
    }


def plot_confusion(matrix: np.ndarray, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    thr = matrix.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > thr else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _display_path(path: Path) -> str:
    """Repo-relative when the file lives in the repo, absolute otherwise.

    Checkpoints normally sit in ``checkpoints/``, but a path outside the repo is
    legitimate (a release download into ``/tmp``, an experiment directory), and
    ``Path.relative_to`` raises instead of falling back.
    """
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/model_epoch_10.pth")
    parser.add_argument("--split", type=Path, default=ROOT / "data/fer2013/test.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results/metrics.json")
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model", default="emotioncnn")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = build_model(args.model)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    params = count_parameters(model)

    dataset = FER2013Dataset(args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    labels, logits = collect_predictions(model, loader, torch.device(args.device))
    metrics = summarize(labels, logits)
    metrics.update({
        "checkpoint": _display_path(args.checkpoint),
        "split": str(args.split),
        "parameters": params,
    })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2))

    print(f"n={metrics['n']}  accuracy={metrics['accuracy']:.4f}  "
          f"macro-F1={metrics['macro_f1']:.4f}  majority-baseline={metrics['majority_class_baseline']:.4f}")
    for name, row in metrics["per_class"].items():
        print(f"  {name:8s} P={row['precision']:.3f} R={row['recall']:.3f} "
              f"F1={row['f1']:.3f}  n={row['support']}")

    if args.figure:
        plot_confusion(np.array(metrics["confusion_matrix"]), args.figure,
                       f"FER2013 - accuracy {metrics['accuracy']:.1%}, macro-F1 {metrics['macro_f1']:.3f}")
        print(f"figure -> {args.figure}")


if __name__ == "__main__":
    main()

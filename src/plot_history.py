"""Plot training history produced by ``src/train.py``.

    python -m src.plot_history results/history_emotioncnn40.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.history.read_text())
    rows = data["history"]
    epochs = [r["epoch"] for r in rows]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax_loss.plot(epochs, [r["train_loss"] for r in rows], label="train")
    ax_loss.plot(epochs, [r["val_loss"] for r in rows], label="val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy")
    ax_loss.set_title("loss")
    ax_loss.legend(frameon=False)

    ax_acc.plot(epochs, [r["val_acc"] * 100 for r in rows], color="tab:green")
    ax_acc.axhline(24.55, ls="--", lw=1, color="grey", label="majority baseline")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("validation accuracy (%)")
    ax_acc.set_title(f"validation accuracy (best {data['best_val_acc']:.1%})")
    ax_acc.legend(frameon=False)

    fig.tight_layout()
    out = args.out or (ROOT / "results/figures" / f"curves_{args.history.stem}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

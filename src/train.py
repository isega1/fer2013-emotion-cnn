"""Train a model and record validation accuracy per epoch.

    python -m src.train --model emotioncnn --epochs 40 --lr 1e-3 --scheduler cosine

Differences from the 2025 coursework script, all deliberate:
  * validation **accuracy** is reported, not only loss;
  * the number of epochs is a flag, not a constant, because the original run
    stopped at 10 while validation accuracy was still climbing;
  * the split is a file on disk (see ``src/split.py``), so a run is
    reproducible instead of depending on a random draw at import time;
  * history is written to JSON next to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import FER2013Dataset
from src.model import build_model, count_parameters

ROOT = Path(__file__).resolve().parents[1]


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_split(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="emotioncnn")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/fer2013")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    model = build_model(args.model).to(device)
    print(f"model={args.model} params={count_parameters(model):,} device={device}")

    train_ds = FER2013Dataset(args.data_dir / "train.csv")
    val_ds = FER2013Dataset(args.data_dir / "val.csv")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if args.scheduler == "cosine" else None)

    history = []
    best_acc = 0.0
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            seen += len(y)
        train_loss = running / seen
        val_loss, val_acc = evaluate_split(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()

        row = {"epoch": epoch, "train_loss": round(train_loss, 4),
               "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4),
               "lr": round(optimizer.param_groups[0]["lr"], 6),
               "seconds": round(time.time() - started, 1)}
        history.append(row)
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss {train_loss:.4f}  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}  ({row['seconds']}s)")

        torch.save(model.state_dict(), args.checkpoint_dir / f"{args.tag}_last.pth")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.checkpoint_dir / f"{args.tag}_best.pth")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.out_dir / f"history_{args.tag}.json"
    history_path.write_text(json.dumps({"args": vars(args) | {"data_dir": str(args.data_dir)},
                                        "best_val_acc": best_acc, "history": history}, indent=2))
    print(f"best val_acc {best_acc:.4f}  history -> {history_path}")


if __name__ == "__main__":
    main()

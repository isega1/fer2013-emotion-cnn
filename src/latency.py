"""CPU latency for several checkpoints, one model per row.

    python -m src.latency --pair cnn-small=checkpoints/m20-cnn-small_best.pth \
        --pair mlp-small=checkpoints/m20-mlp-small_best.pth \
        --out results/latency.json

``src.benchmark`` compares PyTorch against ONNX for a single model; this
script answers the other question, "how long does one 48x48 frame take on a
CPU for each checkpoint", so the 6-model comparison can quote a latency next
to accuracy and parameter count. It reuses ``time_calls`` from
``src.benchmark`` so the numbers are produced by the same timer, with the
same warm-up, thread count and run count as the published 1.71 ms figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.benchmark import time_calls
from src.model import build_model, count_parameters

ROOT = Path(__file__).resolve().parents[1]


def parse_pair(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected MODEL=CHECKPOINT, got {text!r}")
    name, path = text.split("=", 1)
    return name.strip(), Path(path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=parse_pair, action="append", required=True,
                        metavar="MODEL=CHECKPOINT", help="repeat once per checkpoint")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", type=Path, default=ROOT / "results/latency.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    x = torch.randn(1, 1, 48, 48)
    results = {"threads": args.threads, "runs": args.runs, "warmup": args.warmup,
               "batch_size": 1, "models": {}}
    print(f"threads={args.threads}  runs={args.runs}  warmup={args.warmup}  batch=1")
    for name, checkpoint in args.pair:
        model = build_model(name)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            stats = time_calls(lambda: model(x), args.runs, args.warmup)
        results["models"][name] = {
            "checkpoint": checkpoint.name,
            "parameters": count_parameters(model),
            **stats,
        }
        print(f"{name:12s} params {count_parameters(model):>9,d}  "
              f"median {stats['median_ms']:6.2f} ms  p90 {stats['p90_ms']:6.2f} ms")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""CPU latency: PyTorch eager vs ONNX Runtime.

    python -m src.benchmark --checkpoint checkpoints/model_epoch_10.pth \
        --onnx checkpoints/emotioncnn.onnx

Reports the median and 90th percentile of single-batch wall time after a
warm-up, which is what "inference time" should mean. First-call overhead
and thread count are stated rather than hidden: results depend on both.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from src.model import build_model

ROOT = Path(__file__).resolve().parents[1]


def time_calls(fn, runs: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return {
        "median_ms": round(statistics.median(samples), 2),
        "p90_ms": round(samples[int(0.9 * (len(samples) - 1))], 2),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/model_epoch_10.pth")
    parser.add_argument("--onnx", type=Path, default=ROOT / "checkpoints/emotioncnn.onnx")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", type=Path, default=ROOT / "results/benchmark.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    model = build_model("emotioncnn")
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), sess_options=options,
                                   providers=["CPUExecutionProvider"])

    results = {"threads": args.threads, "batch_sizes": {}}
    print(f"threads={args.threads}  runs={args.runs}  warmup={args.warmup}")
    for batch in args.batch_sizes:
        x = torch.randn(batch, 1, 48, 48)

        with torch.no_grad():
            torch_stats = time_calls(lambda: model(x), args.runs, args.warmup)
        np_input = {"input": x.numpy()}
        ort_stats = time_calls(lambda: session.run(None, np_input), args.runs, args.warmup)

        speedup = torch_stats["median_ms"] / ort_stats["median_ms"]
        results["batch_sizes"][batch] = {"torch": torch_stats, "onnxruntime": ort_stats,
                                         "speedup": round(speedup, 2)}
        print(f"batch {batch:2d}  torch {torch_stats['median_ms']:7.2f} ms   "
              f"onnxruntime {ort_stats['median_ms']:7.2f} ms   speedup {speedup:.2f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

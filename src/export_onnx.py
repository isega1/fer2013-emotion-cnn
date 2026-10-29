"""Export a checkpoint to ONNX and verify the export numerically.

    python -m src.export_onnx --checkpoint checkpoints/model_epoch_10.pth \
        --out checkpoints/emotioncnn.onnx

The verification step is the point: torch.onnx.export succeeding proves
nothing on its own, so we run both runtimes on the same random batch and
require the logits to agree to 1e-5.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.model import build_model

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/model_epoch_10.pth")
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints/emotioncnn.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--model", default="emotioncnn")
    args = parser.parse_args()

    model = build_model(args.model)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.randn(1, 1, 48, 48)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        external_data=False,  # single self-contained file, easier to ship
    )
    size_mb = args.out.stat().st_size / 1e6
    print(f"exported {args.out} ({size_mb:.1f} MB, opset {args.opset})")

    import onnxruntime as ort

    session = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    batch = torch.randn(8, 1, 48, 48)
    with torch.no_grad():
        expected = model(batch).numpy()
    got = session.run(None, {"input": batch.numpy()})[0]

    max_diff = float(np.abs(expected - got).max())
    print(f"max |torch - onnxruntime| = {max_diff:.3e}")
    if not np.allclose(expected, got, atol=1e-5):
        raise SystemExit("export verification failed: logits differ by more than 1e-5")
    print("export verified")


if __name__ == "__main__":
    main()

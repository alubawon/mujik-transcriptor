"""Separation benchmark scaffold (v0.4.0)。

跑一次 4-stem Demucs（或 6-stem htdemucs_6s），输出 SDR proxy + 时延。

用法：
    PYTHONPATH=src python benchmarks/run_separation.py --input song.wav --out out.json

v0.4.0 仅 stub：实际 SDR/SIR/SAR 计算留 v0.5（需真实带 ground-truth 的数据集）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Demucs separation and report timing/volume (v0.4.0 scaffold)"
    )
    parser.add_argument("--input", "-i", required=True, help="input audio file")
    parser.add_argument(
        "--variant", choices=["htdemucs_ft", "htdemucs_6s"], default="htdemucs_ft",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "mps"], default="cpu",
    )
    parser.add_argument("--out", "-o", help="output JSON report path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        return 1

    print(f"[bench/separation] input={input_path.name}, variant={args.variant}")
    print(f"[bench/separation] device={args.device}")
    t0 = time.time()
    # v0.4.0 仅记录 metadata；真实 demucs 跑通留 v0.5
    report = {
        "version": "0.4.0-scaffold",
        "input": str(input_path),
        "variant": args.variant,
        "device": args.device,
        "elapsed_sec": 0.0,
        "stems_produced": 0,
        "sdr_db": None,
        "note": "v0.4.0 scaffold only; real SDR/SIR/SAR deferred to v0.5 (requires labeled dataset)",
    }
    elapsed = time.time() - t0
    report["elapsed_sec"] = round(elapsed, 3)

    out_path = Path(args.out) if args.out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[bench/separation] wrote {out_path}")
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Transcription benchmark scaffold (v0.4.0)。

跑一次 basic-pitch / bytedance-piano / adtof，输出 note count + 时延。

用法：
    PYTHONPATH=src python benchmarks/run_transcription.py \\
        --input song.wav --adapter basic-pitch --out out.json

v0.4.0 仅 stub：真实 onset F1 / pitch F1 计算留 v0.5。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run transcription adapter and report timing/note count (v0.4.0 scaffold)"
    )
    parser.add_argument("--input", "-i", required=True, help="input audio file")
    parser.add_argument(
        "--adapter", choices=["basic-pitch", "adtof", "bytedance-piano"],
        default="basic-pitch",
    )
    parser.add_argument("--out", "-o", help="output JSON report path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        return 1

    print(f"[bench/transcription] input={input_path.name}, adapter={args.adapter}")
    t0 = time.time()
    report = {
        "version": "0.4.0-scaffold",
        "input": str(input_path),
        "adapter": args.adapter,
        "elapsed_sec": 0.0,
        "note_count": 0,
        "onset_f1": None,
        "pitch_f1": None,
        "note": "v0.4.0 scaffold only; real F1 deferred to v0.5 (requires labeled dataset)",
    }
    elapsed = time.time() - t0
    report["elapsed_sec"] = round(elapsed, 3)

    out_path = Path(args.out) if args.out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[bench/transcription] wrote {out_path}")
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

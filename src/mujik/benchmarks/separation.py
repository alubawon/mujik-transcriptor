"""分离质量 benchmark（v0.5.2）—— MUSDB18 + museval SDR/SIR/SAR。

对应 docs/research.md §6 清单项 1：Demucs 分离在真实数据上的
SDR/SIR/SAR 评估（SiSEC 标准，museval 实现）。

数据：**仓库不携带 MUSDB18**（research-only 许可）。用户自行下载：
- MUSDB18 压缩版（.mp4 stems，解码需 ffmpeg）
- MUSDB18-HQ（wav，无需 ffmpeg）
  https://sigsep.github.io/datasets/musdb.html

用法：
    .venv/bin/python -m mujik.benchmarks.separation \
        --musdb-root ~/data/musdb18-hq --is-wav \
        --variant htdemucs_ft --device cpu \
        --limit 2 --output sep_bench.md --json sep_bench.json

依赖：pip install 'mujik-transcriptor[separation-bench]'（musdb + museval）。
分离本身复用 separate/separate_audio 路由（v0.5.2 task #1）。
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time
from pathlib import Path

# 评测的 4 个 stem（demucs 4-stem 与 MUSDB18 stem 名一致）
SEPARATION_STEMS: tuple[str, ...] = ("vocals", "drums", "bass", "other")


class SeparationBenchmarkError(RuntimeError):
    """分离 benchmark 依赖或数据缺失。"""


def _require_deps() -> tuple:
    """musdb/museval 延迟 import，缺失 fail-loud 并指向安装命令。

    注意：musdb → stempeg 在 import 时就硬检查 ffmpeg（无则 RuntimeError，
    不是 ImportError），所以 RuntimeError 也要接住并给出安装指引。
    """
    try:
        import musdb
        import museval
    except (ImportError, RuntimeError) as e:
        name = getattr(e, "name", None) or "musdb/museval/ffmpeg"
        raise SeparationBenchmarkError(
            f"missing dependency: {name} — "
            f"run `pip install 'mujik-transcriptor[separation-bench]'`"
            f"（压缩版 MUSDB18 还需系统 ffmpeg；MUSDB18-HQ wav 版不需要）"
        ) from e
    return musdb, museval


def load_musdb(root: str | Path, subset: str = "test", is_wav: bool = False):
    """加载 MUSDB18 数据集（fail-loud：root 不存在直接报错）。"""
    musdb, _ = _require_deps()
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            f"musdb root not found: {root} — "
            f"自行下载 MUSDB18/MUSDB18-HQ（https://sigsep.github.io/datasets/musdb.html），"
            f"仓库不携带数据"
        )
    return musdb.DB(root=str(root), subset=subset, is_wav=is_wav)


def _evaluate_stems(
    references: dict[str, object],
    estimates: dict[str, object],
    museval,
) -> dict[str, dict[str, float]]:
    """museval → per-stem median SDR/SIR/SAR。

    references/estimates: {stem: np.ndarray (nsamples, nch)}。
    堆叠成 (nsrc, nsamples, nch) 后调用 bss_eval_images（返回
    sdr/isr/sir/sar 命名元组，避免 museval.evaluate 数组 metric 顺序歧义）；
    单声道回退 bss_eval_sources（sdr/sir/sar）。
    """
    stems = [s for s in SEPARATION_STEMS if s in references and s in estimates]
    if not stems:
        raise SeparationBenchmarkError("no common stems between reference and estimate")

    import numpy as np
    from museval.metrics import bss_eval_images, bss_eval_sources

    ref = np.stack([references[s] for s in stems])  # (nsrc, n, ch)
    est = np.stack([estimates[s] for s in stems])
    if ref.shape[-1] == 1:  # 单声道 → (nsrc, nsamples)
        sdr, sir, sar, _perm = bss_eval_sources(
            ref[..., 0],
            est[..., 0],
        )
        isr = None
    else:
        sdr, isr, sir, sar, _perm = bss_eval_images(ref, est)
        # 各返回 shape (nsrc, nframes, nchan)

    def _median(vals, i: int) -> float:
        arr = np.asarray(vals, dtype=float)
        arr = arr[i].ravel()
        arr = arr[~np.isnan(arr)]
        return round(float(np.median(arr)), 3) if arr.size else 0.0

    out: dict[str, dict[str, float]] = {}
    for i, stem in enumerate(stems):
        entry = {
            "SDR": _median(sdr, i),
            "SIR": _median(sir, i),
            "SAR": _median(sar, i),
        }
        if isr is not None:
            entry["ISR"] = _median(isr, i)
        out[stem] = entry
    return out


def run_separation_benchmark(
    musdb_root: str | Path,
    subset: str = "test",
    is_wav: bool = False,
    variant: str = "htdemucs_ft",
    device: str = "cpu",
    limit: int | None = None,
    work_dir: str | Path | None = None,
) -> dict:
    """对 MUSDB18 子集逐轨跑 demucs 分离 + museval 评估。

    Returns:
        {
          "tracks": [{track, title, duration_sec, sep_time_sec, scores: {...}}],
          "per_stem_median": {stem: {SDR, SIR, SAR}},
          "mean_sdr": float,          # 4 stem SDR 中位数的均值
        }
    """
    import soundfile as sf

    _, museval = _require_deps()
    from mujik.config.schema import SourceSeparationConfig
    from mujik.separate.router import separate_audio

    tracks = load_musdb(musdb_root, subset=subset, is_wav=is_wav)
    if limit is not None and limit > 0:
        tracks = tracks[:limit]
    if not tracks:
        raise SeparationBenchmarkError(f"no tracks in subset={subset!r}")

    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="mujik_sep_bench_"))
    work.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for idx, track in enumerate(tracks):
        title = f"{track.artist} - {track.title}"
        audio = track.audio  # (nsamples, nch) float64
        mix_path = work / f"mix_{idx:03d}.wav"
        sf.write(mix_path, audio, track.rate)

        sep_cfg = SourceSeparationConfig(
            model="demucs",
            variant=variant,
            device=device,
        )
        out_dir = work / f"sep_{idx:03d}"
        t0 = time.monotonic()
        stems = separate_audio(mix_path, out_dir, config=sep_cfg)
        sep_time = round(time.monotonic() - t0, 1)

        estimates: dict = {}
        for stem_name in SEPARATION_STEMS:
            stem = stems.get(stem_name)
            if stem is None:
                continue
            est, _ = sf.read(stem.audio_path, dtype="float64")
            estimates[stem_name] = est

        references = {
            name: track.targets[name].audio for name in SEPARATION_STEMS if name in track.targets
        }
        scores = _evaluate_stems(references, estimates, museval)

        results.append(
            {
                "track": title,
                "duration_sec": round(len(audio) / track.rate, 1),
                "sep_time_sec": sep_time,
                "scores": scores,
            }
        )
        sdr = scores.get("vocals", {}).get("SDR", float("nan"))
        print(f"[{idx + 1}/{len(tracks)}] {title}: vocals SDR={sdr} ({sep_time}s)")
        mix_path.unlink(missing_ok=True)

    # 聚合：per-stem 跨 track 的 SDR/SIR/SAR 中位数
    per_stem_median: dict[str, dict[str, float]] = {}
    for stem_name in SEPARATION_STEMS:
        per_stem_median[stem_name] = {}
        for metric in ("SDR", "SIR", "SAR"):
            vals = [
                r["scores"][stem_name][metric]
                for r in results
                if stem_name in r["scores"] and metric in r["scores"][stem_name]
            ]
            per_stem_median[stem_name][metric] = round(statistics.median(vals), 3) if vals else 0.0

    mean_sdr = (
        round(
            statistics.mean(v["SDR"] for v in per_stem_median.values() if "SDR" in v),
            3,
        )
        if per_stem_median
        else 0.0
    )

    return {
        "variant": variant,
        "device": device,
        "subset": subset,
        "n_tracks": len(results),
        "tracks": results,
        "per_stem_median": per_stem_median,
        "mean_sdr": mean_sdr,
    }


def render_markdown(report: dict) -> str:
    """分离 benchmark 报告 → markdown。"""
    lines = [
        "# Separation Benchmark (MUSDB18 + museval)",
        "",
        f"- Variant: `{report['variant']}` (device={report['device']})",
        f"- Subset: {report['subset']} / {report['n_tracks']} tracks",
        f"- **Mean SDR (per-stem median): {report['mean_sdr']} dB**",
        "",
        "| Stem | SDR | SIR | SAR |",
        "|---|---|---|---|",
    ]
    for stem, m in report["per_stem_median"].items():
        lines.append(
            f"| {stem} | {m.get('SDR', '—')} | {m.get('SIR', '—')} | {m.get('SAR', '—')} |"
        )
    lines += [
        "",
        "## Per-track",
        "",
        "| Track | Duration | Sep time | vocals SDR | drums SDR | bass SDR | other SDR |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report["tracks"]:
        s = r["scores"]
        row = [r["track"], f"{r['duration_sec']}s", f"{r['sep_time_sec']}s"]
        for stem in SEPARATION_STEMS:
            row.append(str(s.get(stem, {}).get("SDR", "—")))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI：python -m mujik.benchmarks.separation --musdb-root ..."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m mujik.benchmarks.separation",
        description="MUSDB18 分离质量 benchmark（museval SDR/SIR/SAR）",
    )
    parser.add_argument("--musdb-root", required=True, help="MUSDB18/MUSDB18-HQ 解压根目录")
    parser.add_argument("--subset", choices=["train", "test", "all"], default="test")
    parser.add_argument(
        "--is-wav", action="store_true", help="MUSDB18-HQ wav 版（.mp4 压缩版不加，需 ffmpeg）"
    )
    parser.add_argument(
        "--variant",
        default="htdemucs_ft",
        help="demucs variant（htdemucs_ft / htdemucs / htdemucs_6s）",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 轨")
    parser.add_argument(
        "--work-dir", default=None, help="分离产物目录（默认临时目录，评估完保留 stems）"
    )
    parser.add_argument("--output", "-o", default="sep_bench.md")
    parser.add_argument("--json", default="sep_bench.json")
    args = parser.parse_args(argv)

    report = run_separation_benchmark(
        musdb_root=args.musdb_root,
        subset=args.subset,
        is_wav=args.is_wav,
        variant=args.variant,
        device=args.device,
        limit=args.limit,
        work_dir=args.work_dir,
    )

    markdown = render_markdown(report)
    print(markdown)
    Path(args.output).write_text(markdown, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"\nreport → {args.output}" + (f" + {args.json}" if args.json else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "run_separation_benchmark",
    "load_musdb",
    "render_markdown",
    "main",
    "SeparationBenchmarkError",
    "SEPARATION_STEMS",
]

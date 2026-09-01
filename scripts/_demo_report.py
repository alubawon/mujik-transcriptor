#!/usr/bin/env python3
"""scripts/_demo_report.py — demo 产物汇总报告生成器。

v0.5.1 修 5：产物按曲名目录化（demo_out/<曲名>/，多 preset 对比时
demo_out/<曲名>/<preset>/），本脚本递归发现所有 project.json，
以相对路径为行标签输出 markdown 报告。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _summarize_run(run_dir: Path, out_root: Path) -> dict:
    meta = _load_json(run_dir / "project.json") or {}
    # beats/chords 等中间产物在 ws/（多 preset 共享时可能不在本目录下）
    ws = run_dir / "ws"
    if not ws.is_dir():
        candidate = run_dir.parent / "ws"
        ws = candidate if candidate.is_dir() else ws
    beats = _load_json(ws / "beats.json") or {}
    chords = _load_json(ws / "chords.json")
    ts = _load_json(ws / "time_signatures.json") or []
    label = str(run_dir.relative_to(out_root))
    return {
        "run": label,
        "version": meta.get("mujik_version", "?"),
        "preset": meta.get("preset", "?"),
        "separator": meta.get("separator", "?"),
        "transcribe_mode": meta.get("transcribe_mode", "?"),
        "rhythm_enabled": meta.get("rhythm_enabled", False),
        "chord_enabled": meta.get("chord_enabled", False),
        "chord_backend": meta.get("chord_backend", "?"),
        "chord_quantize_enabled": meta.get("chord_quantize_enabled", False),
        "score_features": meta.get("score_features", []),
        "bpm": round(beats["bpm"], 1) if isinstance(beats.get("bpm"), (int, float)) else beats.get("bpm"),
        "n_beats": len(beats.get("beats", [])),
        "n_downbeats": len(beats.get("downbeats", [])),
        "n_chords": len(chords) if chords else 0,
        "n_time_sigs": len(ts),
        "has_mid": (run_dir / "project.mid").exists(),
        "has_musicxml": (run_dir / "score.musicxml").exists(),
        "has_pdf": (run_dir / "score.pdf").exists(),
    }


def _find_run_dirs(out_root: Path) -> list[Path]:
    """递归发现含 project.json 的目录（ws/ 自动排除）。"""
    runs = []
    for meta in sorted(out_root.rglob("project.json")):
        d = meta.parent
        if d.name == "ws":
            continue
        runs.append(d)
    return runs


def render_markdown(out_root: Path) -> str:
    lines: list[str] = []
    lines.append("# mujik-transcriptor demo report\n")
    lines.append(f"Output root: `{out_root}`\n")
    lines.append("")

    runs = _find_run_dirs(out_root)
    summaries = [_summarize_run(d, out_root) for d in runs]

    if not summaries:
        lines.append("_No outputs found (expected: demo_out/<song>/project.json)._\n")
        return "\n".join(lines)

    # 概览表
    lines.append("## Summary\n")
    lines.append("| Run | Preset | Version | Separator | Transcribe | BPM | Beats | Chords | Time-sigs |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| **{s['run']}** | {s['preset']} | {s['version']} | `{s['separator']}` "
            f"| {s['transcribe_mode']} | {s['bpm']} | {s['n_beats']} "
            f"| {s['n_chords']} | {s['n_time_sigs']} |"
        )
    lines.append("")

    # Feature flags
    lines.append("## Feature flags\n")
    lines.append("| Run | Rhythm | Chord | Chord backend | Quantize | Score features |")
    lines.append("|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| **{s['run']}** | {'✅' if s['rhythm_enabled'] else '—'} "
            f"| {'✅' if s['chord_enabled'] else '—'} | `{s['chord_backend']}` "
            f"| {'✅' if s['chord_quantize_enabled'] else '—'} "
            f"| {', '.join(s['score_features']) or '—'} |"
        )
    lines.append("")

    # Artifacts
    lines.append("## Artifacts\n")
    lines.append("| Run | MIDI | MusicXML | PDF |")
    lines.append("|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| **{s['run']}** "
            f"| {'✅' if s['has_mid'] else '❌'} "
            f"| {'✅' if s['has_musicxml'] else '❌'} "
            f"| {'✅' if s['has_pdf'] else '❌'} |"
        )
    lines.append("")

    # Per-run detail
    lines.append("## Per-run detail\n")
    for s in summaries:
        lines.append(f"### {s['run']}\n")
        lines.append("```json")
        lines.append(json.dumps(
            {k: v for k, v in s.items() if not k.startswith("has_")},
            indent=2, ensure_ascii=False,
        ))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: _demo_report.py <demo_out_dir>", file=sys.stderr)
        return 2
    out_root = Path(sys.argv[1])
    if not out_root.is_dir():
        print(f"not a dir: {out_root}", file=sys.stderr)
        return 2
    print(render_markdown(out_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())

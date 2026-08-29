#!/usr/bin/env python3
"""scripts/_demo_report.py — 汇总三 preset 输出的报告生成器。

读取 demo_out/{pop,jazz,metal}/ 下的 project.json + beats.json + chords.json，
输出一个可读的 markdown 报告到 stdout。
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


def _summarize_preset(preset_dir: Path) -> dict:
    meta = _load_json(preset_dir / "project.json") or {}
    beats = _load_json(preset_dir / "beats.json") or {}
    chords = _load_json(preset_dir / "chords.json")
    ts = _load_json(preset_dir / "time_signatures.json") or []
    return {
        "preset": preset_dir.name,
        "version": meta.get("mujik_version", "?"),
        "separator": meta.get("separator", "?"),
        "transcribe_mode": meta.get("transcribe_mode", "?"),
        "rhythm_enabled": meta.get("rhythm_enabled", False),
        "chord_enabled": meta.get("chord_enabled", False),
        "chord_backend": meta.get("chord_backend", "?"),
        "chord_quantize_enabled": meta.get("chord_quantize_enabled", False),
        "score_features": meta.get("score_features", []),
        "bpm": beats.get("bpm"),
        "n_beats": len(beats.get("beats", [])),
        "n_downbeats": len(beats.get("downbeats", [])),
        "n_chords": len(chords) if chords else 0,
        "n_time_sigs": len(ts),
        "has_mid": (preset_dir / "project.mid").exists(),
        "has_musicxml": (preset_dir / "score.musicxml").exists(),
        "has_pdf": (preset_dir / "score.pdf").exists(),
    }


def render_markdown(out_root: Path) -> str:
    lines: list[str] = []
    lines.append("# mujik-transcriptor demo report\n")
    lines.append(f"Output root: `{out_root}`\n")
    lines.append("")

    presets = ["pop", "jazz", "metal"]
    summaries = []
    for p in presets:
        d = out_root / p
        if d.exists():
            summaries.append(_summarize_preset(d))

    if not summaries:
        lines.append("_No preset outputs found._\n")
        return "\n".join(lines)

    # 概览表
    lines.append("## Summary\n")
    lines.append("| Preset | Version | Separator | Transcribe | BPM | Beats | Chords | Time-sigs |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| **{s['preset']}** | {s['version']} | `{s['separator']}` "
            f"| {s['transcribe_mode']} | {s['bpm']} | {s['n_beats']} "
            f"| {s['n_chords']} | {s['n_time_sigs']} |"
        )
    lines.append("")

    # Feature flags
    lines.append("## Feature flags\n")
    lines.append("| Preset | Rhythm | Chord | Chord backend | Quantize | Score features |")
    lines.append("|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| **{s['preset']}** | {'✅' if s['rhythm_enabled'] else '—'} "
            f"| {'✅' if s['chord_enabled'] else '—'} | `{s['chord_backend']}` "
            f"| {'✅' if s['chord_quantize_enabled'] else '—'} "
            f"| {', '.join(s['score_features']) or '—'} |"
        )
    lines.append("")

    # Artifacts
    lines.append("## Artifacts\n")
    lines.append("| Preset | MIDI | MusicXML | PDF |")
    lines.append("|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| **{s['preset']}** "
            f"| {'✅' if s['has_mid'] else '❌'} "
            f"| {'✅' if s['has_musicxml'] else '❌'} "
            f"| {'✅' if s['has_pdf'] else '❌'} |"
        )
    lines.append("")

    # Per-preset detail
    lines.append("## Per-preset detail\n")
    for s in summaries:
        lines.append(f"### {s['preset']}\n")
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

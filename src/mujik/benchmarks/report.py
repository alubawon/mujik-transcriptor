"""Benchmark report generator（v0.5.0）。

BenchmarkReport → markdown 表格 + JSON dump。
"""
from __future__ import annotations

import json

from mujik.benchmarks import BenchmarkReport


_PRIMARY_SCORE: dict[str, str] = {
    "note_transcription": "F1",
    "beat_tracking": "CMLt",
    "chord_recognition": "majmin",
}


def render_markdown(report: BenchmarkReport) -> str:
    """生成 markdown 报告。

    包含：
    - 顶部 summary（version / dataset / n_samples / overall）
    - per-genre 表格（5 genre × 3 metric）
    - per-sample 详情（可选）
    """
    lines: list[str] = []
    lines.append(f"# Benchmark Report (v{report.version})")
    lines.append("")
    lines.append(f"- Dataset: `{report.dataset_name}`")
    lines.append(f"- Samples: {report.n_samples}")
    lines.append("")

    # Overall
    lines.append("## Overall")
    lines.append("")
    lines.append("| Metric | Primary Score |")
    lines.append("|---|---|")
    for metric, primary in _PRIMARY_SCORE.items():
        score = report.overall.get(metric, "—")
        lines.append(f"| {metric} | {score} |")
    lines.append("")

    # Per-genre
    lines.append("## Per-Genre")
    lines.append("")
    lines.append("| Genre | " + " | ".join(_PRIMARY_SCORE.values()) + " |")
    lines.append("|---|" + "---|" * len(_PRIMARY_SCORE))
    for genre in sorted(report.per_genre.keys()):
        scores = report.per_genre[genre]
        row = [genre]
        for metric in _PRIMARY_SCORE:
            score = scores.get(metric, "—")
            row.append(str(score))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-sample（v0.5.2：含 n_pred/n_gt——管线崩溃记的 0 分与真实 0 分
    # 在此可区分：前者 n_pred=0 且有 warning traceback 在 stderr/JSON）
    lines.append("## Per-Sample")
    lines.append("")
    lines.append("| Sample | Genre | note n_pred/n_gt | note F1 | beat CMLt | chord majmin |")
    lines.append("|---|---|---|---|---|---|")
    for sm in report.per_sample:
        nt = sm.metrics.get("note_transcription", {})
        bt = sm.metrics.get("beat_tracking", {})
        cr = sm.metrics.get("chord_recognition", {})
        lines.append(
            f"| {sm.sample_id} | {sm.genre} "
            f"| {nt.get('n_pred', '—')}/{nt.get('n_gt', '—')} "
            f"| {nt.get('f1', '—')} | {bt.get('cmlt', '—')} | {cr.get('majmin', '—')} |"
        )
    lines.append("")

    return "\n".join(lines)


def render_json(report: BenchmarkReport) -> str:
    """生成 JSON 报告（per_sample + per_genre + overall）。"""
    return json.dumps({
        "version": report.version,
        "dataset_name": report.dataset_name,
        "n_samples": report.n_samples,
        "overall": report.overall,
        "per_genre": report.per_genre,
        "per_sample": [
            {
                "sample_id": sm.sample_id,
                "genre": sm.genre,
                "metrics": sm.metrics,
            }
            for sm in report.per_sample
        ],
    }, ensure_ascii=False, indent=2)


__all__ = ["render_markdown", "render_json"]

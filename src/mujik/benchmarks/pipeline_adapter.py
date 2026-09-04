"""真实管线 benchmark adapter（v0.5.2）。

把 v0.2.2 完整 Pipeline 接进 BenchmarkRunner 的 pipeline_func 契约：

    PipelineBenchmarkAdapter(...)(audio_path) → {
        "note_transcription": {"notes": [(pitch, onset, offset), ...]},
        "beat_tracking":      {"beats": [float, ...]},
        "chord_recognition":  {"chords": [(start, end, root, quality), ...]},
    }

每样本在 work_dir/<stem>/ 下跑一次完整管线（中间产物留在 ws/），
predicted 全部取自 Project 对象与 beats.json artifact——
benchmark 度量的是真实全栈（demucs + madmom + basic-pitch/drumscript + chord），
不是 mock。

pipeline 异常不在此吞掉：交给 BenchmarkRunner 的失败处理（该样本记 0 分）。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from mujik.benchmarks import BenchmarkSample  # noqa: F401  (契约说明)
from mujik.config.schema import PipelineConfig

# apply_preset 支持的 preset（custom 无需 apply）
_PRESETS = ("pop", "jazz", "metal")


class PipelineBenchmarkAdapter:
    """跑真实 mujik 管线，产出 benchmark runner 契约的 predicted dict。

    Args:
        preset: pop / jazz / metal（走 apply_preset）
        work_dir: 每样本管线的落盘根目录（样本输出在 work_dir/<stem>/）
        enable_chords: 强制开 chord 检测（benchmark 需要和弦指标；
            pop/metal preset 默认关 chord，这里强制开）
    """

    def __init__(
        self,
        preset: str = "pop",
        work_dir: str | Path = "bench_work",
        enable_chords: bool = True,
    ):
        if preset not in _PRESETS:
            raise ValueError(f"preset must be one of {_PRESETS}, got {preset!r}")
        self.preset = preset
        self.work_dir = Path(work_dir)
        self.enable_chords = enable_chords

    def __call__(self, audio_path: str | Path) -> dict:
        from mujik.pipeline import Pipeline

        audio_path = Path(audio_path)
        sample_out = self.work_dir / audio_path.stem
        ws_dir = sample_out / "ws"

        cfg = PipelineConfig(
            input_path=str(audio_path),
            output_dir=str(sample_out),
            workspace_dir=str(ws_dir),
            preset=self.preset,
        )
        if self.preset in _PRESETS:
            cfg = cfg.apply_preset(self.preset)
        if self.enable_chords:
            cfg.chord.enabled = True

        logger.info(
            "benchmark pipeline: input={}, preset={}, out={}",
            audio_path,
            self.preset,
            sample_out,
        )
        project = Pipeline(cfg).run()

        # notes：全部 tracks 合并（benchmark 只看音符层面，不分轨）
        pred_notes = [
            (n.pitch, n.start, n.end) for track in project.tracks.values() for n in track.notes
        ]
        pred_notes.sort(key=lambda t: (t[1], t[0]))

        # beats：ws/beats.json（madmom artifact）
        # v0.5.2: 缺文件或 rhythm 关闭时打 warning——此前静默空列表会让
        # beat CMLt 全 0 却看不出原因
        pred_beats: list[float] = []
        beats_json = ws_dir / "beats.json"
        if beats_json.is_file():
            pred_beats = [float(b) for b in json.loads(beats_json.read_text()).get("beats", [])]
        else:
            logger.warning(
                "benchmark pipeline: beats.json not found at %s "
                "(rhythm disabled?) — beat CMLt will be 0",
                beats_json,
            )

        # chords：project.chord_track（quantize/groove 后，v0.4.9）
        pred_chords = [(c.start, c.end, c.root, c.quality) for c in (project.chord_track or [])]

        logger.info(
            "benchmark pipeline: {} notes / {} beats / {} chords",
            len(pred_notes),
            len(pred_beats),
            len(pred_chords),
        )
        return {
            "note_transcription": {"notes": pred_notes},
            "beat_tracking": {"beats": pred_beats},
            "chord_recognition": {"chords": pred_chords},
        }


def _build_default_metric_calculators() -> dict[str, Callable]:
    """3 个 metric calculator（延迟 import，无 mir_eval 时降级逻辑在 metrics 内）。"""
    from mujik.benchmarks.metrics import (
        BeatTrackingMetrics,
        ChordRecognitionMetrics,
        NoteTranscriptionMetrics,
    )

    return {
        "note_transcription": NoteTranscriptionMetrics(),
        "beat_tracking": BeatTrackingMetrics(),
        "chord_recognition": ChordRecognitionMetrics(),
    }


__all__ = ["PipelineBenchmarkAdapter", "_build_default_metric_calculators"]

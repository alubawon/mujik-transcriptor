"""Benchmark metrics（v0.5.0）。

3 个 metric calculator：
- NoteTranscriptionMetrics: F1 / Precision / Recall（onset ±50ms 容忍）
- BeatTrackingMetrics: CMLt / AMLt（用 mir_eval.beat）
- ChordRecognitionMetrics: majmin / root / sevenths（用 mir_eval.chord）

依赖：
- mir_eval（已在 pyproject.toml）
- 无需 pandas
"""
from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)


# ---------- Note Transcription ----------

class NoteTranscriptionMetrics:
    """Note 转录 F1 / Precision / Recall。

    predicted/ground_truth 格式：
        {"notes": [(pitch:int, onset:float, offset:float), ...]}

    onset 容忍：±50ms（音乐转录常用）
    pitch 必须严格相等
    """

    ONSET_TOLERANCE_SEC = 0.05  # 50ms

    @property
    def name(self) -> str:
        return "note_transcription"

    def compute(self, predicted: dict, ground_truth: dict) -> dict[str, float]:
        pred_notes = predicted.get("notes", [])
        gt_notes = ground_truth.get("notes", [])

        n_pred = len(pred_notes)
        n_gt = len(gt_notes)
        if n_pred == 0 and n_gt == 0:
            return {"f1": 1.0, "precision": 1.0, "recall": 1.0, "n_pred": 0, "n_gt": 0}
        if n_pred == 0:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "n_pred": 0, "n_gt": n_gt}
        if n_gt == 0:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "n_pred": n_pred, "n_gt": 0}

        # 匹配：onset 差 ≤ 50ms 且 pitch 相等
        matched_gt = [False] * n_gt
        tp = 0
        for p_pitch, p_onset, _ in pred_notes:
            for i, (g_pitch, g_onset, _) in enumerate(gt_notes):
                if matched_gt[i]:
                    continue
                if p_pitch == g_pitch and abs(p_onset - g_onset) <= self.ONSET_TOLERANCE_SEC:
                    matched_gt[i] = True
                    tp += 1
                    break

        precision = tp / n_pred
        recall = tp / n_gt
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "n_pred": n_pred,
            "n_gt": n_gt,
            "n_tp": tp,
        }


# ---------- Beat Tracking ----------

class BeatTrackingMetrics:
    """节拍跟踪 CMLt / AMLt。

    predicted/ground_truth 格式：
        {"beats": [float, ...]}  # seconds
    """

    @property
    def name(self) -> str:
        return "beat_tracking"

    def compute(self, predicted: dict, ground_truth: dict) -> dict[str, float]:
        pred_beats = predicted.get("beats", [])
        gt_beats = ground_truth.get("beats", [])

        if not gt_beats:
            # ground truth 为空 → metrics 不可计算
            return {"cmlt": 0.0, "amlt": 0.0, "n_pred": len(pred_beats), "n_gt": 0}

        try:
            import mir_eval
            scores = mir_eval.beat.evaluate(
                np.asarray(gt_beats),
                np.asarray(pred_beats),
            )
            # scores: dict with CMLt, AMLt, P-score, etc.
            cmlt = float(scores.get("CMLt", 0.0))
            amlt = float(scores.get("AMLt", 0.0))
            return {
                "cmlt": round(cmlt, 4),
                "amlt": round(amlt, 4),
                "n_pred": len(pred_beats),
                "n_gt": len(gt_beats),
            }
        except ImportError:
            logger.warning("mir_eval not available, beat metrics = 0")
            return {"cmlt": 0.0, "amlt": 0.0, "n_pred": len(pred_beats), "n_gt": len(gt_beats)}


# ---------- Chord Recognition ----------

class ChordRecognitionMetrics:
    """和弦识别 majmin / root / sevenths（mir_eval.chord）。

    predicted/ground_truth 格式：
        {"chords": [(start, end, root, quality), ...]}
    """

    @property
    def name(self) -> str:
        return "chord_recognition"

    def compute(self, predicted: dict, ground_truth: dict) -> dict[str, float]:
        pred_chords = predicted.get("chords", [])
        gt_chords = ground_truth.get("chords", [])

        if not gt_chords:
            return {"majmin": 0.0, "root": 0.0, "sevenths": 0.0,
                    "n_pred": len(pred_chords), "n_gt": 0}

        # 转换为 mir_eval 格式：(intervals [n,2] float, labels list[str])
        # v0.5.1 修：mir_eval.chord.evaluate 签名是
        # (ref_intervals, ref_labels, est_intervals, est_labels) 四参数，
        # 原实现只传 2 个数组——dev 镜像未装 mir_eval 走 ImportError 分支，
        # 一直没暴露（ml 镜像装 mir_eval 0.8.2 后 TypeError）
        # mir_eval 只认完整 quality 名（min/maj7/min7...），不认 "m" 缩写
        _Q2MEVAL = {
            "": "maj", "maj": "maj", "M": "maj",
            "m": "min", "min": "min",
            "dim": "dim", "aug": "aug",
            "7": "7", "maj7": "maj7",
            "m7": "min7", "min7": "min7",
            "dim7": "dim7", "hdim7": "hdim7", "m7b5": "m7b5",
            "sus2": "sus2", "sus4": "sus4",
            "6": "6", "maj6": "maj6", "m6": "min6", "min6": "min6",
            "9": "9", "maj9": "maj9", "m9": "min9", "min9": "min9",
            "11": "11", "13": "13",
        }

        def to_meval(chords):
            intervals = np.array(
                [(s, e) for s, e, _, _ in chords], dtype=float,
            ).reshape(-1, 2)
            labels = [
                f"{r}:{_Q2MEVAL.get(q, q)}" for _, _, r, q in chords
            ]
            return intervals, labels

        try:
            import mir_eval
            ref_iv, ref_lb = to_meval(gt_chords)
            est_iv, est_lb = to_meval(pred_chords)
            scores = mir_eval.chord.evaluate(ref_iv, ref_lb, est_iv, est_lb)
            # scores: dict with 'root', 'thirds', 'triads', 'sevenths', 'majmin', 'mirex'
            return {
                "majmin": round(float(scores.get("majmin", 0.0)), 4),
                "root": round(float(scores.get("root", 0.0)), 4),
                "sevenths": round(float(scores.get("sevenths", 0.0)), 4),
                "n_pred": len(pred_chords),
                "n_gt": len(gt_chords),
            }
        except ImportError:
            logger.warning("mir_eval not available, chord metrics = 0")
            return {"majmin": 0.0, "root": 0.0, "sevenths": 0.0,
                    "n_pred": len(pred_chords), "n_gt": len(gt_chords)}


# 延迟 import numpy（mir_eval 需要）
import numpy as np


__all__ = [
    "NoteTranscriptionMetrics",
    "BeatTrackingMetrics",
    "ChordRecognitionMetrics",
]

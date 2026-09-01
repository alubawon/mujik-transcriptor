"""Synthetic 5-genre baseline dataset（v0.5.0）。

设计动机：CI smoke 验证 benchmark framework 不依赖外部数据集。
程序生成 5 genre × 3 file = 15 个 5s 测试样本：
- 已知 ground truth（chord progression + beat grid + note melody）
- 5 genre 用不同 base 频率范围区分（pop 220Hz / jazz 110Hz / metal 82Hz / rnb 165Hz / classical 261Hz）

注意：synthetic 音频不是真实音乐，仅用于验证 framework + sanity check。
真实评估需 v0.5.1+ 引入 MusicNet/MAPS/BALLROOM 等数据集。
"""
from __future__ import annotations

import json
import tempfile
import wave
from pathlib import Path
from typing import ClassVar

import numpy as np

from mujik.benchmarks import BENCHMARK_GENRES, BenchmarkSample


# 5 genre 的 base 频率（Hz）— 区分性音调
_GENRE_FREQS: dict[str, float] = {
    "pop": 220.0,      # A3
    "jazz": 110.0,     # A2
    "metal": 82.41,    # E2
    "rnb": 165.0,      # E3
    "classical": 261.63,  # C4
}

# 5 genre 的 chord progression（每 bar 一个 chord，4 bar = 16 beats）
_GENRE_PROGRESSIONS: dict[str, list[str]] = {
    "pop": ["C", "G", "Am", "F"],
    "jazz": ["Dm7", "G7", "Cmaj7", "A7"],
    "metal": ["Em", "G", "D", "A"],
    "rnb": ["Am7", "Dm7", "G7", "Cmaj7"],
    "classical": ["C", "F", "G", "C"],
}


def _generate_synthetic_wav(
    path: Path,
    genre: str,
    bpm: float = 120.0,
    bars: int = 4,
    sample_rate: int = 22050,
) -> float:
    """生成 synthetic 音频 + 写入 .wav。

    Returns:
        duration in seconds
    """
    freq = _GENRE_FREQS[genre]
    beat_dur = 60.0 / bpm
    bar_dur = beat_dur * 4
    duration = bar_dur * bars
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # base 频率 + 简单和声（5度 + 8度）
    audio = (
        0.3 * np.sin(2 * np.pi * freq * t)
        + 0.15 * np.sin(2 * np.pi * freq * 1.5 * t)  # 5度
        + 0.1 * np.sin(2 * np.pi * freq * 2.0 * t)   # 8度
    ).astype(np.float32)

    # 写到 wav（16-bit PCM）
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_int16 = np.int16(audio * 32767)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return duration


def _parse_chord_name(name: str) -> tuple[str, str]:
    """解析 lead-sheet 风格 chord 名 → (root, quality)。

    v0.5.1 修：原解析对 "Dm7"/"Cmaj7" 会切出 root="Dm"/"Cmaj"
    （root 混入了 quality 字符，mir_eval 无法编码）。
    长后缀优先匹配："Cmaj7"→("C","maj7"), "Dm7"→("D","m7"),
    "G7"→("G","7"), "Am"→("A","m"), "C"→("C","")。
    """
    for suffix, quality in (
        ("maj7", "maj7"), ("m7", "m7"), ("7", "7"), ("m", "m"),
    ):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)], quality
    return name, ""


def _generate_ground_truth(
    genre: str,
    bpm: float,
    duration: float,
) -> dict:
    """生成已知 ground truth：notes / beats / chords。"""
    beat_dur = 60.0 / bpm
    progression = _GENRE_PROGRESSIONS[genre]

    # beats: 每 beat 一个
    beats = [i * beat_dur for i in range(int(duration / beat_dur) + 1) if i * beat_dur < duration]

    # chords: 每 bar 一个
    chords = []
    for i, ch in enumerate(progression):
        start = i * beat_dur * 4
        end = min((i + 1) * beat_dur * 4, duration)
        root, quality = _parse_chord_name(ch)
        chords.append((start, end, root, quality))

    # notes: 1 个 melody note per beat (频率随 chord 变化)
    base_freq = _GENRE_FREQS[genre]
    notes = []
    for i, onset in enumerate(beats[:-1]):
        pitch = int(round(69 + 12 * np.log2(base_freq * (440 / 220) * (1 + 0.1 * (i % 4)) / 440)))
        notes.append((pitch, onset, onset + beat_dur * 0.5))

    return {"notes": notes, "beats": beats, "chords": chords}


class SyntheticBenchmarkDataset:
    """5 genre × 3 file = 15 个 synthetic 测试样本。

    单例：首次调用 generate() 时生成所有文件到临时目录。
    后续 list_samples() 返回缓存的样本列表。
    """

    SAMPLES_PER_GENRE: ClassVar[int] = 3
    BPM: ClassVar[float] = 120.0
    BARS: ClassVar[int] = 4

    def __init__(self, base_dir: str | Path | None = None):
        if base_dir is None:
            base_dir = Path(tempfile.mkdtemp(prefix="mujik_bench_synth_"))
        else:
            base_dir = Path(base_dir)
            base_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir = base_dir
        self._samples: list[BenchmarkSample] | None = None

    @property
    def name(self) -> str:
        return "synthetic_5genre_baseline"

    def list_samples(self) -> list[BenchmarkSample]:
        if self._samples is not None:
            return self._samples

        samples: list[BenchmarkSample] = []
        for genre in BENCHMARK_GENRES:
            for i in range(self.SAMPLES_PER_GENRE):
                sample_id = f"{genre}_{i + 1:02d}"
                audio_path = self.base_dir / f"{sample_id}.wav"
                gt_path = self.base_dir / f"{sample_id}.json"

                # 生成 wav（如果不存在）
                if not audio_path.exists():
                    duration = _generate_synthetic_wav(
                        audio_path, genre, bpm=self.BPM, bars=self.BARS,
                    )
                    gt = _generate_ground_truth(genre, self.BPM, duration)
                    gt_path.write_text(json.dumps(gt), encoding="utf-8")
                else:
                    duration = self.BPM and (self.BPM * self.BARS * 4 / 60.0)
                    gt = json.loads(gt_path.read_text(encoding="utf-8"))

                samples.append(BenchmarkSample(
                    sample_id=sample_id,
                    genre=genre,
                    audio_path=str(audio_path),
                    duration=duration,
                    gt_notes=gt.get("notes", []),
                    gt_beats=gt.get("beats", []),
                    gt_chords=gt.get("chords", []),
                ))
        self._samples = samples
        return samples


__all__ = ["SyntheticBenchmarkDataset"]

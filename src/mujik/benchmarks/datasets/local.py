"""本地真实数据集 adapter（v0.5.2）。

解决真实数据 benchmark 的版权问题：**仓库不携带任何数据**，
用户在自家曲库目录放一份 manifest.json + 音频文件即可评测。

目录结构（manifest 中路径相对 data_dir）：
    my_bench/
    ├── manifest.json
    └── audio/
        ├── song1.wav
        └── song2.mp3

manifest.json 格式（顶层 list 或 {"samples": [...]} 均可）：
    [
      {
        "sample_id": "buhee_30s",
        "genre": "jazz",
        "audio": "audio/buhee_30s.wav",
        "notes":  [[60, 0.5, 1.2], ...],          # 可选 [pitch, onset, offset]
        "beats":  [0.0, 0.5, 1.0, ...],           # 可选（秒）
        "chords": [[0.0, 2.0, "C", "maj7"], ...]  # 可选 [start, end, root, quality]
      },
      ...
    ]

设计原则（fail-loud）：
- manifest 缺失 / 音频缺失 / 必填字段缺失 / 格式错误 → 直接抛异常，
  绝不静默跳过（静默会算出假 0 分，比报错更有害）。
"""

from __future__ import annotations

import json
from pathlib import Path

from mujik.benchmarks import BenchmarkSample

MANIFEST_NAME = "manifest.json"


class LocalDatasetError(RuntimeError):
    """本地数据集 manifest 或音频不合法。"""


def _validate_notes(raw, sample_id: str) -> list[tuple[int, float, float]]:
    notes: list[tuple[int, float, float]] = []
    for i, item in enumerate(raw):
        if not (isinstance(item, (list, tuple)) and len(item) == 3):
            raise LocalDatasetError(
                f"{sample_id}: notes[{i}] 必须是 [pitch, onset, offset]，得到 {item!r}"
            )
        pitch, onset, offset = item
        if not (isinstance(pitch, int) and 0 <= pitch <= 127):
            raise LocalDatasetError(
                f"{sample_id}: notes[{i}] pitch 必须是 0-127 整数，得到 {pitch!r}"
            )
        notes.append((int(pitch), float(onset), float(offset)))
    return notes


def _validate_beats(raw, sample_id: str) -> list[float]:
    beats: list[float] = []
    for i, item in enumerate(raw):
        if not isinstance(item, (int, float)):
            raise LocalDatasetError(f"{sample_id}: beats[{i}] 必须是秒数，得到 {item!r}")
        beats.append(float(item))
    return beats


def _validate_chords(raw, sample_id: str) -> list[tuple[float, float, str, str]]:
    chords: list[tuple[float, float, str, str]] = []
    for i, item in enumerate(raw):
        if not (isinstance(item, (list, tuple)) and len(item) == 4):
            raise LocalDatasetError(
                f"{sample_id}: chords[{i}] 必须是 [start, end, root, quality]，得到 {item!r}"
            )
        start, end, root, quality = item
        if not (isinstance(root, str) and root):
            raise LocalDatasetError(
                f"{sample_id}: chords[{i}] root 必须是非空字符串，得到 {root!r}"
            )
        chords.append((float(start), float(end), str(root), str(quality or "")))
    return chords


class LocalBenchmarkDataset:
    """manifest 驱动的本地数据集 adapter。

    Usage:
        ds = LocalBenchmarkDataset("~/my_bench")
        samples = ds.list_samples()
    """

    def __init__(self, data_dir: str | Path, manifest_name: str = MANIFEST_NAME):
        self.data_dir = Path(data_dir).expanduser().resolve()
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")
        self.manifest_path = self.data_dir / manifest_name
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"manifest not found: {self.manifest_path} — "
                f"真实数据 benchmark 需要用户在 data_dir 提供 manifest.json"
            )
        self._samples: list[BenchmarkSample] | None = None

    @property
    def name(self) -> str:
        return f"local:{self.data_dir.name}"

    def _load_manifest(self) -> list[dict]:
        raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("samples")
        if not isinstance(raw, list) or not raw:
            raise LocalDatasetError(
                f'{self.manifest_path}: 顶层必须是非空 list 或 {{"samples": [...]}}'
            )
        return raw

    def list_samples(self) -> list[BenchmarkSample]:
        if self._samples is not None:
            return self._samples

        samples: list[BenchmarkSample] = []
        for i, entry in enumerate(self._load_manifest()):
            if not isinstance(entry, dict):
                raise LocalDatasetError(f"manifest[{i}] 必须是 object，得到 {type(entry).__name__}")
            missing = [k for k in ("sample_id", "genre", "audio") if not entry.get(k)]
            if missing:
                raise LocalDatasetError(
                    f"manifest[{i}] 缺少必填字段: {missing} (必填 sample_id/genre/audio)"
                )
            sample_id = str(entry["sample_id"])
            genre = str(entry["genre"])
            audio_path = self.data_dir / str(entry["audio"])
            if not audio_path.is_file():
                raise FileNotFoundError(f"{sample_id}: audio not found: {audio_path}")

            notes = _validate_notes(entry.get("notes", []), sample_id)
            beats = _validate_beats(entry.get("beats", []), sample_id)
            chords = _validate_chords(entry.get("chords", []), sample_id)

            samples.append(
                BenchmarkSample(
                    sample_id=sample_id,
                    genre=genre,
                    audio_path=str(audio_path),
                    duration=float(entry.get("duration", 0.0) or 0.0),
                    gt_notes=notes,
                    gt_beats=beats,
                    gt_chords=chords,
                )
            )
        self._samples = samples
        return samples


__all__ = ["LocalBenchmarkDataset", "LocalDatasetError", "MANIFEST_NAME"]

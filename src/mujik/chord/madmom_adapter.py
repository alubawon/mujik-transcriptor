"""madmom 和弦识别 adapter（v0.4.4，subprocess 隔离）。

madmom（BSD-3）通过 ``CRFChordRecognitionProcessor``（CNN 特征 + CRF 解码）
检测音频中的和弦，输出 (start, end, chord_label) 三元组。

subprocess 模式与 rhythm adapter 一致：写临时 wrapper 脚本 → 调 madmom →
写 JSON → 读回。**不直接 import madmom 包**到主线 → 避免触发 liccheck 警告。

**已知限制**（madmom CRNN 模型固有）：
- 仅 25 类：12 maj + 12 min + 1 no-chord
- **不支持** 7th（maj7/min7/dom7）、延伸（9/11/13）、sus/dim/aug
- 7th / 扩展和弦 → v0.4.5+ 用 BTC-HCQT 或 Chord-CNN-LSTM

输出 JSON 格式：
```json
[
  {"start": 0.5, "end": 2.0, "label": "C:maj"},
  {"start": 2.0, "end": 4.0, "label": "F:maj"},
  {"start": 4.0, "end": 4.5, "label": "N"},
  ...
]
```

Label 字符串格式：
- ``"C:maj"`` / ``"F#:min"`` / ``"Bb:maj"`` — 正常和弦
- ``"N"`` — no-chord（silence / 弱信号）
- ``"X"`` — unknown（低置信度）
- 其他异常格式 → 跳过（不抛错）
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from mujik.midi.model import ChordEvent

if TYPE_CHECKING:
    from mujik.config.schema import ChordConfig


# madmom chord detection subprocess 默认超时（30 分钟，足以处理 5 分钟音频）
MADMOM_CHORD_TIMEOUT_DEFAULT = 1800


class MadmomChordAdapterError(RuntimeError):
    pass


# madmom chord label → 内部 quality 字符串映射
# madmom 输出 "maj" / "min"；我们用 "" / "m"（与 ChordEvent.QUALITY_TO_KIND 对齐）
_MADMOM_QUALITY_MAP: dict[str, str] = {
    "maj": "",
    "major": "",
    "M": "",
    "min": "m",
    "minor": "m",
    "m": "m",
    "-": "m",
}


_MADMOM_CHORD_WRAPPER = r'''
"""madmom chord recognition wrapper.

Usage: _madmom_chord_wrapper.py <input_audio> <output_json>

Exit codes:
  2 = usage error
  3 = madmom not installed
  4 = audio load failed
  5 = chord detection failed
"""
import json
import sys


def main():
    if len(sys.argv) != 3:
        print("usage: _madmom_chord_wrapper.py <input> <output_json>", file=sys.stderr)
        sys.exit(2)

    audio_path = sys.argv[1]
    json_path = sys.argv[2]

    try:
        from madmom.io.audio import load_audio_file
        from madmom.features.chords import (
            CNNChordFeatureProcessor,
            CRFChordRecognitionProcessor,
        )
    except ImportError:
        print("madmom not installed; install via `uv pip install madmom`", file=sys.stderr)
        sys.exit(3)

    try:
        sig, _sr = load_audio_file(audio_path)
    except Exception as e:
        print(f"failed to load audio: {e}", file=sys.stderr)
        sys.exit(4)

    try:
        # CNN 特征 → 25-bin chroma (12 maj + 12 min + 1 N)
        feat_proc = CNNChordFeatureProcessor()
        feats = feat_proc(sig)
        # CRF Viterbi 解码 → (start, end, label) structured array
        chord_proc = CRFChordRecognitionProcessor()
        chords = chord_proc(feats)
    except Exception as e:
        print(f"chord detection failed: {e}", file=sys.stderr)
        sys.exit(5)

    out = []
    for entry in chords:
        # madmom 0.16+ 返回 dtype [('start', '<f8'), ('end', '<f8'), ('chord', 'S32')]
        start = float(entry[0])
        end = float(entry[1])
        raw_label = entry[2]
        # 兼容 S32 bytes 与 str 两种返回类型
        if isinstance(raw_label, bytes):
            label = raw_label.decode("utf-8")
        else:
            label = str(raw_label)
        out.append({"start": start, "end": end, "label": label})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
'''


def check_madmom_chord_available() -> bool:
    """检查 madmom 是否可 import（与 rhythm adapter 相同的检查模式）。"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import madmom"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _parse_madmom_chord_label(label: str) -> ChordEvent | None:
    """madmom chord label 字符串 → ChordEvent 或 None（skip）。

    解析规则：
    - ``"N"`` / ``"X"`` → None（无和弦 / 未知）
    - ``"Root:quality"`` → split on first ``":"`` → root + quality_short
    - quality_short 通过 ``_MADMOM_QUALITY_MAP`` 标准化
      - ``"maj"`` / ``"major"`` / ``"M"`` → ``""``（major，无后缀）
      - ``"min"`` / ``"minor"`` / ``"m"`` / ``"-"`` → ``"m"``
      - 其他 → 透传（v0.4.4 不会触发，CRNN 模型只输出 maj/min）
    - 缺 ``":"`` 或 root 异常 → None（跳过）

    start/end 暂时用 0.0 占位，调用方在循环中填充。

    Examples:
        >>> _parse_madmom_chord_label("C:maj")
        ChordEvent(start=0.0, end=0.0, root="C", quality="")
        >>> _parse_madmom_chord_label("F#:min")
        ChordEvent(start=0.0, end=0.0, root="F#", quality="m")
        >>> _parse_madmom_chord_label("N") is None
        True
        >>> _parse_madmom_chord_label("X") is None
        True
    """
    label = label.strip()
    if not label or label in ("N", "X"):
        return None

    if ":" not in label:
        logger.warning("madmom_chord: skipping malformed label: {label!r}", label=label)
        return None

    root, _, quality_short = label.partition(":")
    root = root.strip()
    quality_short = quality_short.strip()
    if not root or not quality_short:
        return None

    quality = _MADMOM_QUALITY_MAP.get(quality_short, quality_short)
    return ChordEvent(start=0.0, end=0.0, root=root, quality=quality)


def detect_chords_with_madmom(
    audio_path: str | Path,
    config: "ChordConfig | None" = None,
    out_dir: str | Path | None = None,
) -> list[ChordEvent]:
    """用 madmom CRNN 检测和弦 → ChordEvent 列表。

    Args:
        audio_path: 输入 wav 路径
        config: ChordConfig（v0.4.4 仅使用 ``chord_timeout_sec``；``models`` /
            ``vocab`` 字段为未来 BTC-HCQT / Chord-CNN 预留）
        out_dir: 输出目录；None 时写到 tmp

    Returns:
        list[ChordEvent]：madmom 输出的和弦事件（``N`` / ``X`` 已过滤），
        按 start 升序

    Raises:
        FileNotFoundError: 音频文件不存在
        MadmomChordAdapterError: subprocess 失败 / 超时 / 输出解析失败
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    timeout = (
        config.chord_timeout_sec
        if config is not None and hasattr(config, "chord_timeout_sec")
        else MADMOM_CHORD_TIMEOUT_DEFAULT
    )

    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="mujik_chord_"))
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"chords_{audio_path.stem}.json"
    # v0.5.1 修 5：wrapper 脚本写系统临时目录，不再泄漏进产物目录
    wrapper_path = Path(tempfile.gettempdir()) / f"mujik_madmom_chord_wrapper_{os.getpid()}.py"
    wrapper_path.write_text(_MADMOM_CHORD_WRAPPER, encoding="utf-8")

    logger.info(
        "madmom_chord: input={input}, timeout={sec}s",
        input=audio_path, sec=timeout,
    )

    cmd = [sys.executable, str(wrapper_path), str(audio_path), str(json_path)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise MadmomChordAdapterError(
            f"madmom chord detection timeout after {timeout}s"
        ) from e

    # cleanup wrapper（best-effort）
    try:
        wrapper_path.unlink()
    except OSError:
        pass

    if result.returncode != 0:
        raise MadmomChordAdapterError(
            f"madmom chord failed (exit={result.returncode}): {result.stderr[:500]}"
        )

    if not json_path.exists():
        raise MadmomChordAdapterError(
            f"madmom chord output json not found: {json_path}"
        )

    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise MadmomChordAdapterError(
            f"failed to parse madmom chord output: {e}"
        ) from e

    chord_track: list[ChordEvent] = []
    for entry in raw:
        try:
            label = entry["label"]
            start = float(entry["start"])
            end = float(entry["end"])
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(
                "madmom_chord: skipping malformed entry {entry}: {e}",
                entry=entry, e=e,
            )
            continue
        chord = _parse_madmom_chord_label(label)
        if chord is None:
            continue
        chord.start = start
        chord.end = end
        chord_track.append(chord)

    logger.info(
        "madmom_chord: {n} chords detected from {audio}",
        n=len(chord_track), audio=audio_path.name,
    )
    return chord_track


__all__ = [
    "MADMOM_CHORD_TIMEOUT_DEFAULT",
    "MadmomChordAdapterError",
    "check_madmom_chord_available",
    "detect_chords_with_madmom",
]

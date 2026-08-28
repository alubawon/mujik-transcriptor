"""Spotify basic-pitch 转录 adapter（subprocess 隔离）。

basic-pitch 是基于 TensorFlow 的多音转录库（Apache-2.0），通过 CLI 调用：
    basic-pitch <output_dir> <input_audio>

输出 CSV 列：[start_time_s, end_time_s, pitch_midi, pitch_velocity[, pitch_bend]]

为什么 subprocess：TF 依赖重（~500MB），与主进程 PyTorch 栈混合会有冲突，
design.md §8 明确要求 TF 走子进程。
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from loguru import logger

from mujik.config.schema import BasicPitchConfig
from mujik.midi.model import Note

BASIC_PITCH_CLI = "basic-pitch"


class BasicPitchAdapterError(RuntimeError):
    pass


def check_basic_pitch_available() -> bool:
    """检查 basic-pitch CLI 是否在 PATH 中。"""
    try:
        result = subprocess.run(
            [BASIC_PITCH_CLI, "--help"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def transcribe_with_basic_pitch(
    audio_path: str | Path,
    config: BasicPitchConfig | None = None,
    out_dir: str | Path | None = None,
) -> list[Note]:
    """调用 basic-pitch CLI 转写音频 → list[Note]。

    Args:
        audio_path: 输入音频（WAV/FLAC/MP3）
        config: basic-pitch 配置
        out_dir: basic-pitch 输出目录；None 时用输入文件同目录

    Returns:
        list[Note]：所有 onset 事件，已按 start 排序
    """
    cfg = config or BasicPitchConfig()
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    if out_dir is None:
        out_dir = audio_path.parent
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # basic-pitch CLI: basic-pitch <out_dir> <input> [flags]
    cmd = [
        BASIC_PITCH_CLI,
        str(out_dir),
        str(audio_path),
    ]
    # 透传配置
    cmd.extend(["--onset-threshold", str(cfg.onset_threshold)])
    cmd.extend(["--frame-threshold", str(cfg.frame_threshold)])
    cmd.extend(["--min-note-length", str(int(cfg.min_note_length_ms))])
    if cfg.min_frequency is not None:
        cmd.extend(["--min-frequency", str(cfg.min_frequency)])
    if cfg.max_frequency is not None:
        cmd.extend(["--max-frequency", str(cfg.max_frequency)])

    logger.info(
        "basic-pitch: input={input}, out_dir={out}, "
        "onset={onset}, frame={frame}, min_len={ms}ms",
        input=audio_path, out=out_dir,
        onset=cfg.onset_threshold, frame=cfg.frame_threshold,
        ms=cfg.min_note_length_ms,
    )

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=cfg.timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        raise BasicPitchAdapterError(
            f"basic-pitch timeout after {cfg.timeout_sec}s"
        ) from e

    if result.returncode != 0:
        raise BasicPitchAdapterError(
            f"basic-pitch failed (exit={result.returncode}): "
            f"{result.stderr[:500]}"
        )

    # 解析输出：<input_stem>_basic_pitch.csv
    csv_path = out_dir / f"{audio_path.stem}_basic_pitch.csv"
    if not csv_path.exists():
        raise BasicPitchAdapterError(
            f"basic-pitch output csv not found: {csv_path}; "
            f"out_dir contents: {list(out_dir.iterdir())}"
        )

    notes: list[Note] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = float(row["start_time_s"])
                end = float(row["end_time_s"])
                pitch = int(row["pitch_midi"])
                velocity = int(round(float(row["pitch_velocity"])))
            except (KeyError, ValueError) as e:
                logger.warning("basic-pitch: skip malformed row {}: {}", row, e)
                continue

            if not (0 <= pitch <= 127):
                logger.warning("basic-pitch: skip out-of-range pitch={}", pitch)
                continue
            if not (0 <= velocity <= 127):
                velocity = max(0, min(127, velocity))

            # v0.4.0: 解析 pitch_bend 列（JSON list of floats in [-1, 1]）
            pitch_bend: tuple[float, ...] = ()
            bend_str = row.get("pitch_bend", "").strip()
            if bend_str:
                try:
                    import json
                    bend_list = json.loads(bend_str)
                    if isinstance(bend_list, list):
                        # 校验每个值
                        pitch_bend = tuple(
                            max(-1.0, min(1.0, float(v))) for v in bend_list
                        )
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.debug("basic-pitch: skip pitch_bend parse: {}", e)

            notes.append(Note(
                start=start,
                end=max(end, start + 0.01),  # 至少 10ms
                pitch=pitch,
                velocity=velocity,
                channel=0,  # pitched stem 用 channel 0
                pitch_bend=pitch_bend,
            ))

    notes.sort(key=lambda n: n.start)
    logger.info(
        "basic-pitch: {n} notes ({b} with pitch_bend)",
        n=len(notes), b=sum(1 for x in notes if x.pitch_bend),
    )
    return notes


__all__ = [
    "transcribe_with_basic_pitch",
    "check_basic_pitch_available",
    "BasicPitchAdapterError",
    "BASIC_PITCH_CLI",
]

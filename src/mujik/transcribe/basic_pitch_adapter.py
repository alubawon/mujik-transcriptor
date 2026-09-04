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

# bend 去噪参数（v0.5.3）
# basic-pitch 的 pitch_bend 是逐帧 semitone 整数，大量 note 带全量逐帧抖动
# （buhee 实测 66484 个 bend 事件 vs 6090 note）——绝大多数是模型噪声而非
# 演奏细节，全量渲染会撑爆 MusicXML/PDF
_MIN_BEND_SEGMENT_FRAMES = 3   # 单个连续段至少 3 帧才算有意 bend
_MIN_BEND_TOTAL_FRAMES = 4     # 去噪后总帧数不足则整个丢弃


def denoise_bend(
    values: list[float] | tuple[float, ...],
    min_segment_frames: int = _MIN_BEND_SEGMENT_FRAMES,
    min_total_frames: int = _MIN_BEND_TOTAL_FRAMES,
) -> tuple[float, ...]:
    """逐帧 bend 序列去噪。

    步骤：
      1. 去掉首尾的 0 段（"无 bend" 帧）
      2. 合并连续同值段
      3. 丢弃短于 min_segment_frames 的段（逐帧抖动）
      4. 剩余总帧数 < min_total_frames → 整个丢弃（视为噪声）
      5. 恒定值序列（全程同一非零值）也丢弃——那是音高识别偏差而非 bend，
         渲染成静态 bend 只会把音准画歪

    Returns:
        去噪后的逐帧 bend（可能为空 tuple）。
    """
    if not values:
        return ()

    # 1. strip 首尾 0
    lo, hi = 0, len(values)
    while lo < hi and abs(values[lo]) < 1e-9:
        lo += 1
    while hi > lo and abs(values[hi - 1]) < 1e-9:
        hi -= 1
    core = values[lo:hi]
    if not core:
        return ()

    # 2. 连续同值段
    segments: list[tuple[float, int]] = []  # (value, count)
    for v in core:
        if segments and abs(segments[-1][0] - v) < 1e-9:
            segments[-1] = (segments[-1][0], segments[-1][1] + 1)
        else:
            segments.append((v, 1))

    # 恒定值（单段）→ 音高偏差而非 bend
    if len(segments) == 1:
        return ()

    # 3. 丢短段；全被丢掉也返回空
    kept = [(v, c) for v, c in segments if c >= min_segment_frames]
    if not kept:
        return ()

    # 4. 总帧数下限
    total = sum(c for _, c in kept)
    if total < min_total_frames:
        return ()

    # 展开回逐帧（保留被保留段的原始重复）
    out: list[float] = []
    for v, c in kept:
        out.extend([v] * c)
    return tuple(out)


class BasicPitchAdapterError(RuntimeError):
    pass


def resolve_basic_pitch_cli() -> str:
    """解析 basic-pitch CLI 路径。

    优先取当前解释器同目录的 console script（venv 下直接跑 `.venv/bin/mujik`
    时子进程 PATH 里未必有 .venv/bin）；不存在再回退裸命令名（依赖 PATH，
    容器/系统安装场景）。
    """
    sibling = Path(sys.executable).parent / BASIC_PITCH_CLI
    if sibling.is_file():
        return str(sibling)
    return BASIC_PITCH_CLI


def check_basic_pitch_available() -> bool:
    """检查 basic-pitch CLI 是否可用（venv 同目录或 PATH）。"""
    try:
        result = subprocess.run(
            [resolve_basic_pitch_cli(), "--help"],
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
    # v0.5.1 修：basic-pitch ≥0.3 的 flag 是全称（--minimum-note-length 等，
    # 原 --min-note-length 等缩写不存在 → exit 2 usage error）；
    # 且 --save-note-events 默认 False，不加则不产出 adapter 要解析的 csv
    cmd = [
        resolve_basic_pitch_cli(),
        "--save-note-events",
        str(out_dir),
        str(audio_path),
    ]
    # 透传配置
    cmd.extend(["--onset-threshold", str(cfg.onset_threshold)])
    cmd.extend(["--frame-threshold", str(cfg.frame_threshold)])
    cmd.extend(["--minimum-note-length", str(int(cfg.min_note_length_ms))])
    if cfg.min_frequency is not None:
        cmd.extend(["--minimum-frequency", str(cfg.min_frequency)])
    if cfg.max_frequency is not None:
        cmd.extend(["--maximum-frequency", str(cfg.max_frequency)])

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
                # v0.5.1 修：basic-pitch ≥0.3 列名是 velocity（原 pitch_velocity）
                velocity = int(round(float(row["velocity"])))
            except (KeyError, ValueError) as e:
                logger.warning("basic-pitch: skip malformed row {}: {}", row, e)
                continue

            if not (0 <= pitch <= 127):
                logger.warning("basic-pitch: skip out-of-range pitch={}", pitch)
                continue
            if not (0 <= velocity <= 127):
                velocity = max(0, min(127, velocity))

            # v0.5.1 修：basic-pitch ≥0.3 的 pitch_bend 是逐帧 semitone 整数
            # 以逗号续在同一行（DictReader 把多出来的列放进 row[None]），
            # 不再是 JSON list。semitone → mujik 的 [-1,+1] 满量程：除以 2
            # （默认 bend range ±2 semitones）
            # v0.5.3: 逐帧 bend 去噪（抖动段/恒定值丢弃），见 denoise_bend
            pitch_bend: tuple[float, ...] = ()
            bend_values = [row.get("pitch_bend", ""), *(row.get(None) or [])]
            bend_values = [v for v in bend_values if v not in (None, "")]
            if bend_values:
                try:
                    pitch_bend = denoise_bend([
                        max(-1.0, min(1.0, float(v) / 2.0)) for v in bend_values
                    ])
                except (ValueError, TypeError) as e:
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
    "resolve_basic_pitch_cli",
    "denoise_bend",
    "BasicPitchAdapterError",
    "BASIC_PITCH_CLI",
]

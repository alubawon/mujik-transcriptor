"""adtof 鼓转录 adapter（subprocess 隔离）。

adtof 是 PyTorch-based 鼓转录库（MIT），输出 onset events。
通过 subprocess 调用避免污染主进程，且与 demucs / basic-pitch 模式一致。

输出格式：subprocess 写一个 CSV [time_s, class_id, velocity]
本模块解析 CSV → list[Note]（channel 9, GM drum note number）。

GM 标准鼓映射（5-class）：
  0 (kick)       → 36 (Bass Drum 1)
  1 (snare)      → 38 (Acoustic Snare)
  2 (closed hh)  → 42 (Closed Hi-Hat)
  3 (open hh)    → 46 (Open Hi-Hat)
  4 (cymbal)     → 49 (Crash Cymbal 1)

GM 标准鼓映射（9-class）：
  0 kick  1 snare  2 hi-hat closed  3 hi-hat open
  4 tom-hi  5 tom-mid  6 tom-low
  7 crash  8 ride
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

from mujik.config.schema import AdtofConfig
from mujik.midi.io import DRUM_CHANNEL
from mujik.midi.model import Note

# adtof class index → GM 标准鼓 note number
GM_DRUM_MAP_5CLASS: dict[int, int] = {
    0: 36,  # Bass Drum 1
    1: 38,  # Acoustic Snare
    2: 42,  # Closed Hi-Hat
    3: 46,  # Open Hi-Hat
    4: 49,  # Crash Cymbal 1
}

GM_DRUM_MAP_9CLASS: dict[int, int] = {
    0: 36,  # Bass Drum 1
    1: 38,  # Acoustic Snare
    2: 42,  # Closed Hi-Hat
    3: 46,  # Open Hi-Hat
    4: 50,  # High Tom
    5: 47,  # Low-Mid Tom
    6: 45,  # Low Tom
    7: 49,  # Crash Cymbal 1
    8: 51,  # Ride Cymbal 1
}

GM_DRUM_MAPS: dict[str, dict[int, int]] = {
    "adtof-5class": GM_DRUM_MAP_5CLASS,
    "adtof-9class": GM_DRUM_MAP_9CLASS,
}

# adtof 调用脚本（写进临时文件后 subprocess 执行）
_ADTOF_WRAPPER = r'''
"""adtof 调用 wrapper：参数 <input_audio> <output_csv> [model] [device] [threshold]"""
import sys
import csv

def main():
    if len(sys.argv) < 3:
        print("usage: _adtof_wrapper.py <input> <output_csv> [model] [device] [threshold]", file=sys.stderr)
        sys.exit(2)
    input_path = sys.argv[1]
    output_csv = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "adtof-5class"
    device = sys.argv[4] if len(sys.argv) > 4 else "cpu"
    threshold = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5

    try:
        from adtof.model.pytorch.predict import predict
    except ImportError:
        print("adtof not installed; install via `uv pip install adtof`", file=sys.stderr)
        sys.exit(3)

    # predict(audio_path) → 2D array [time, class, velocity]
    try:
        events = predict(input_path, model=model, device=device)
    except TypeError:
        # 旧版 API：只接受 audio_path
        events = predict(input_path)

    # 写 CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "class_id", "velocity"])
        for ev in events:
            # ev = (time, class_id, velocity) or [time, class, velocity]
            t, c, v = float(ev[0]), int(ev[1]), float(ev[2])
            if v < threshold:
                continue
            writer.writerow([t, c, v])

if __name__ == "__main__":
    main()
'''


class AdtofAdapterError(RuntimeError):
    pass


def check_adtof_available() -> bool:
    """检查 adtof 是否在 venv 中可用。"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import adtof"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def transcribe_drums_with_adtof(
    audio_path: str | Path,
    config: AdtofConfig | None = None,
    out_dir: str | Path | None = None,
) -> list[Note]:
    """调用 adtof 转录音频为鼓事件 → Note 列表（固定 channel 9）。

    Args:
        audio_path: 输入音频（WAV/FLAC/MP3）
        config: adtof 配置
        out_dir: 子进程临时输出目录；None 时用系统 temp

    Returns:
        list[Note]：每个 onset 一个 Note；channel 固定 9，duration = min_note_length_ms / 1000
    """
    cfg = config or AdtofConfig()
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    if out_dir is None:
        out_dir = Path(tempfile.gettempdir())
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"adtof_{audio_path.stem}.csv"
    # v0.5.1 修 5：wrapper 脚本写系统临时目录，不再泄漏进产物目录
    wrapper_path = Path(tempfile.gettempdir()) / f"mujik_adtof_wrapper_{os.getpid()}.py"
    wrapper_path.write_text(_ADTOF_WRAPPER)

    duration_sec = cfg.min_note_length_ms / 1000.0

    logger.info(
        "adtof: input={input}, model={model}, device={device}, threshold={thr}",
        input=audio_path, model=cfg.model, device=cfg.device, thr=cfg.onset_threshold,
    )

    cmd = [
        sys.executable, str(wrapper_path),
        str(audio_path), str(csv_path),
        cfg.model, cfg.device, str(cfg.onset_threshold),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=cfg.timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        raise AdtofAdapterError(
            f"adtof timeout after {cfg.timeout_sec}s"
        ) from e

    if result.returncode != 0:
        raise AdtofAdapterError(
            f"adtof failed (exit={result.returncode}): {result.stderr[:500]}"
        )

    if not csv_path.exists():
        raise AdtofAdapterError(
            f"adtof output csv not found: {csv_path}"
        )

    drum_map = GM_DRUM_MAPS.get(cfg.model, GM_DRUM_MAP_5CLASS)
    notes: list[Note] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row["time_s"])
                class_id = int(row["class_id"])
                velocity = float(row["velocity"])
            except (KeyError, ValueError) as e:
                logger.warning("adtof: skip malformed row {}: {}", row, e)
                continue

            pitch = drum_map.get(class_id)
            if pitch is None:
                logger.warning("adtof: unknown class_id={}, skip", class_id)
                continue

            vel_int = int(round(min(max(velocity, 0.0), 1.0) * 127))
            if vel_int < 1:
                continue

            notes.append(Note(
                start=t,
                end=t + duration_sec,
                pitch=pitch,
                velocity=vel_int,
                channel=DRUM_CHANNEL,
            ))

    # 清理 wrapper（CSV 留给调用方调试）
    try:
        wrapper_path.unlink()
    except OSError:
        pass

    notes.sort(key=lambda n: n.start)
    logger.info("adtof: {n} drum events", n=len(notes))
    return notes


__all__ = [
    "transcribe_drums_with_adtof",
    "check_adtof_available",
    "AdtofAdapterError",
    "GM_DRUM_MAP_5CLASS",
    "GM_DRUM_MAP_9CLASS",
]

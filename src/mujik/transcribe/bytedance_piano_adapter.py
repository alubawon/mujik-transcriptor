"""ByteDance piano_transcription_inference adapter（v0.4.0）。

subprocess 模式：写 wrapper 脚本 → 调 `python -m piano_transcription_inference` → 解析输出 MIDI。

设计决策（v0.4.0）：
- 镜像不烧 PyTorch（与 adtof/madmom 同模式）
- 单独 extra `[transcribe-bytedance]`，用户自装
- 模型权重首次运行自动下载到 ~/.piano_transcription_inference/
- 输出 MIDI 用 pretty_midi 解析

约定：
- 输入：wav 路径
- 输出：list[Note]，仅 note（pedal track 丢弃）
- 默认 channel=0（piano pitched）
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

from mujik.config.schema import TranscribeConfig
from mujik.midi.model import Note


PIANO_TRANSCRIPTION_MODULE = "piano_transcription_inference"
BYTEDANCE_TIMEOUT_DEFAULT = 1800


class ByteDancePianoAdapterError(RuntimeError):
    pass


def check_bytedance_piano_available() -> bool:
    """检查 `piano_transcription_inference` 模块是否可 import。"""
    try:
        __import__(PIANO_TRANSCRIPTION_MODULE)
        return True
    except ImportError:
        return False


def _write_wrapper(input_path: Path, output_midi_path: Path) -> Path:
    """写 tmp wrapper 脚本（subprocess 调 `python <wrapper>`）。"""
    wrapper_code = f'''"""Auto-generated wrapper for piano_transcription_inference (v0.4.0)."""
import sys
import json


def main():
    input_path = sys.argv[1]
    output_midi_path = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"

    try:
        from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
    except ImportError as e:
        print(f"ERROR: piano_transcription_inference not installed: {{e}}", file=sys.stderr)
        sys.exit(2)

    try:
        audio, _ = load_audio(input_path, sr=sample_rate, mono=True)
        transcriptor = PianoTranscription(device=device, checkpoint_path=None)
        transcriptor.transcribe(audio, sample_rate, output_midi_path)
    except Exception as e:
        print(f"ERROR: transcription failed: {{e}}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
'''
    fd, path = tempfile.mkstemp(suffix="_bytedance_piano.py", prefix="mujik_")
    with open(fd, "w") as f:
        f.write(wrapper_code)
    return Path(path)


def transcribe_piano_bytedance(
    audio_path: Path | str,
    config: TranscribeConfig | None = None,
    out_dir: Path | str | None = None,
    device: str = "cpu",
) -> list[Note]:
    """用 ByteDance piano_transcription_inference 转写钢琴音频。

    Args:
        audio_path: 输入音频
        config: TranscribeConfig（暂未用）
        out_dir: 输出目录；None 时写到 tmp
        device: cuda / cpu

    Returns:
        list[Note]：所有 note（pedal track 过滤）
    """
    audio_path = Path(audio_path)

    if not check_bytedance_piano_available():
        raise ByteDancePianoAdapterError(
            "piano_transcription_inference not installed; "
            "install via `uv pip install 'mujik-transcriptor[transcribe-bytedance]'`"
        )

    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="mujik_bytedance_"))
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    output_midi = out_dir / f"{audio_path.stem}_bytedance.mid"

    wrapper = _write_wrapper(audio_path, output_midi)
    cmd = [sys.executable, str(wrapper), str(audio_path), str(output_midi), device]
    logger.info(
        "bytedance piano: cmd={cmd}",
        cmd=" ".join(cmd),
    )
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=BYTEDANCE_TIMEOUT_DEFAULT,
        )
    except subprocess.TimeoutExpired as e:
        # 清理 wrapper
        try:
            wrapper.unlink()
        except FileNotFoundError:
            pass
        raise ByteDancePianoAdapterError(
            f"bytedance piano timeout ({BYTEDANCE_TIMEOUT_DEFAULT}s)"
        ) from e
    finally:
        try:
            wrapper.unlink()
        except FileNotFoundError:
            pass

    if result.returncode != 0:
        raise ByteDancePianoAdapterError(
            f"bytedance piano failed (exit={result.returncode}): "
            f"stderr={result.stderr[:500]}"
        )

    if not output_midi.exists():
        raise ByteDancePianoAdapterError(
            f"bytedance piano output midi not found: {output_midi}"
        )

    # 用 pretty_midi 解析输出
    try:
        import pretty_midi
    except ImportError as e:
        raise ByteDancePianoAdapterError(
            f"pretty-midi required to parse bytedance output: {e}"
        ) from e

    pm = pretty_midi.PrettyMIDI(str(output_midi))
    notes: list[Note] = []
    for inst in pm.instruments:
        # ByteDance 输出通常有 2 个 track：note + pedal。pedal 名字含 "pedal"
        if "pedal" in inst.name.lower():
            continue
        for n in inst.notes:
            notes.append(Note(
                start=float(n.start),
                end=float(n.end),
                pitch=int(n.pitch),
                velocity=int(n.velocity),
                channel=0,
            ))

    notes.sort(key=lambda n: n.start)
    logger.info(
        "bytedance piano: {n} notes from {audio}",
        n=len(notes), audio=audio_path.name,
    )
    return notes


__all__ = [
    "transcribe_piano_bytedance",
    "check_bytedance_piano_available",
    "ByteDancePianoAdapterError",
    "PIANO_TRANSCRIPTION_MODULE",
]

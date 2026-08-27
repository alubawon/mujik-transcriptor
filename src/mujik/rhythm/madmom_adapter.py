"""madmom 节拍/下拍/BPM 跟踪 adapter（subprocess 隔离）。

madmom（BSD-3）通过 RNNDownBeatProcessor + DownBeatTrackingProcessor 提取
beat/downbeat，用 TempoEstimationProcessor 估全局 BPM。

subprocess 模式与 adtof 一致：写临时 wrapper 脚本 → 调 madmom → 写 JSON → 读回。

输出 JSON 格式：
{
  "beats": [t0, t1, ...],
  "downbeats": [t0, t1, ...],
  "bpm": 120.0,
  "tempo_confidence": 0.85
}
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from loguru import logger

from mujik.config.schema import RhythmConfig
from mujik.rhythm.model import BeatTrack


class MadmomAdapterError(RuntimeError):
    pass


_MADMON_WRAPPER = r'''
"""madmom 调用 wrapper：<input_audio> <output_json>"""
import sys
import json

def main():
    if len(sys.argv) < 3:
        print("usage: _madmom_wrapper.py <input> <output_json>", file=sys.stderr)
        sys.exit(2)
    input_path = sys.argv[1]
    output_json = sys.argv[2]

    try:
        import numpy as np
        from madmom.io.audio import load_audio_file
        from madmom.features.beats import RNNDownBeatProcessor, DownBeatTrackingProcessor
        from madmom.features.tempo import TempoEstimationProcessor
    except ImportError:
        print("madmom not installed; install via `uv pip install madmom`", file=sys.stderr)
        sys.exit(3)

    # 加载音频（madmom 自己读文件）
    try:
        sig, sr = load_audio_file(input_path)
    except Exception as e:
        print(f"failed to load audio: {e}", file=sys.stderr)
        sys.exit(4)

    # RNN 下打概率
    rnn = RNNDownBeatProcessor()
    probs = rnn(sig)

    # 下打跟踪
    db_proc = DownBeatTrackingProcessor(beats_per_bar=[3, 4, 5, 6, 7])
    db_result = db_proc(probs)  # [(time, label), ...]  label 1=beat, 0=downbeat

    downbeats = [float(t) for t, lab in db_result if lab == 0]
    beats = [float(t) for t, _ in db_result]

    # 估计全局 BPM
    tempo_proc = TempoEstimationProcessor(fps=100)
    tempi = tempo_proc(probs)
    # tempi 是 list of (bpm, strength)，取 strength 最高
    if tempi:
        bpm, strength = max(tempi, key=lambda x: x[1])
        bpm = float(bpm)
        # 强度 ∈ [0, 1]，作为置信度
        tempo_confidence = min(1.0, float(strength) / 100.0)
    else:
        bpm, tempo_confidence = 120.0, 0.0

    out = {
        "beats": beats,
        "downbeats": downbeats,
        "bpm": bpm,
        "tempo_confidence": tempo_confidence,
    }
    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
'''


def check_madmom_available() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import madmom"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def track_beats_with_madmom(
    audio_path: str | Path,
    config: RhythmConfig | None = None,
    out_dir: str | Path | None = None,
) -> BeatTrack:
    """调用 madmom 提取 beats / downbeats / tempo → BeatTrack。"""
    cfg = config or RhythmConfig()
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    if out_dir is None:
        out_dir = Path("/tmp")
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"madmom_{audio_path.stem}.json"
    wrapper_path = out_dir / "_madmom_wrapper.py"
    wrapper_path.write_text(_MADMON_WRAPPER)

    logger.info(
        "madmom: input={input}, timeout={sec}s",
        input=audio_path, sec=cfg.madmom_timeout_sec,
    )

    cmd = [sys.executable, str(wrapper_path), str(audio_path), str(json_path)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=cfg.madmom_timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        raise MadmomAdapterError(
            f"madmom timeout after {cfg.madmom_timeout_sec}s"
        ) from e

    if result.returncode != 0:
        raise MadmomAdapterError(
            f"madmom failed (exit={result.returncode}): {result.stderr[:500]}"
        )

    if not json_path.exists():
        raise MadmomAdapterError(
            f"madmom output json not found: {json_path}"
        )

    data = json.loads(json_path.read_text())
    try:
        wrapper_path.unlink()
    except OSError:
        pass

    track = BeatTrack(
        beats=data.get("beats", []),
        downbeats=data.get("downbeats", []),
        bpm=float(data.get("bpm", 120.0)),
        tempo_confidence=float(data.get("tempo_confidence", 0.0)),
    )

    logger.info(
        "madmom: {n} beats, {d} downbeats, bpm={b:.1f} (conf={c:.2f})",
        n=len(track.beats), d=len(track.downbeats),
        b=track.bpm, c=track.tempo_confidence,
    )
    return track


__all__ = [
    "track_beats_with_madmom",
    "check_madmom_available",
    "MadmomAdapterError",
]

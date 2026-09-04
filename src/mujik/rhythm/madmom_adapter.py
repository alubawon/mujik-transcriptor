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
import os
import subprocess
import sys
import tempfile
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
        # madmom 0.16 的 Cython 扩展（ml/hmm.pyx 编译产物 .so）运行时引用
        # np.int / np.float，numpy 1.24+ 已移除；.so 无法像 .py 一样 sed 补丁，
        # 这里 monkey-patch 回内建类型（等价替换，官方推荐做法）
        for _name, _builtin in (
            ("int", int), ("float", float), ("complex", complex),
            ("object", object), ("bool", bool),
        ):
            if not hasattr(np, _name):
                setattr(np, _name, _builtin)
        from madmom.io.audio import load_audio_file
        # madmom 0.16: RNNDownBeatProcessor / DBNDownBeatTrackingProcessor 在
        # madmom.features.downbeats（v0.5.1 修：原 import 自 features.beats 会
        # ImportError 且被 except ImportError 误报为 "not installed"）
        from madmom.features.downbeats import (
            RNNDownBeatProcessor,
            DBNDownBeatTrackingProcessor,
        )
        from madmom.features.tempo import TempoEstimationProcessor
    except ImportError as e:
        print(f"madmom import failed: {e}", file=sys.stderr)
        print("madmom not installed; install via `uv pip install madmom`", file=sys.stderr)
        sys.exit(3)

    # madmom 0.16 + numpy>=1.24 兼容补丁：DBNDownBeatTrackingProcessor.process
    # 里 `np.argmax(np.asarray(results)[:, 1])` 对 ragged (path, log_prob) 列表
    # 会 ValueError（numpy 1.24 起禁止隐式 object array）。
    # 复制原方法并只替换 best 选择一行（其余逐行同 madmom 0.16 源码）。
    def _patched_dbn_process(self, activations, **kwargs):
        import itertools as it
        from madmom.features.downbeats import _process_dbn
        first = 0
        if self.threshold:
            idx = np.nonzero(activations >= self.threshold)[0]
            if idx.any():
                first = max(first, np.min(idx))
                last = min(len(activations), np.max(idx) + 1)
            else:
                last = first
            activations = activations[first:last]
        if not activations.any():
            return np.empty((0, 2))
        results = list(self.map(_process_dbn, zip(self.hmms,
                                                  it.repeat(activations))))
        # v0.5.1 修：按 log probability 选最佳 HMM（原 np.asarray ragged 崩溃点）
        best = max(range(len(results)), key=lambda i: results[i][1])
        path, _ = results[best]
        st = self.hmms[best].transition_model.state_space
        om = self.hmms[best].observation_model
        positions = st.state_positions[path]
        beat_numbers = positions.astype(int) + 1
        if self.correct:
            beats = np.empty(0, dtype=int)
            beat_range = om.pointers[path] >= 1
            idx = np.nonzero(np.diff(beat_range.astype(int)))[0] + 1
            if beat_range[0]:
                idx = np.r_[0, idx]
            if beat_range[-1]:
                idx = np.r_[idx, beat_range.size]
            if idx.any():
                for left, right in idx.reshape((-1, 2)):
                    peak = np.argmax(activations[left:right]) // 2 + left
                    beats = np.hstack((beats, peak))
        else:
            beats = np.nonzero(np.diff(beat_numbers))[0] + 1
        return np.vstack(((beats + first) / float(self.fps),
                          beat_numbers[beats])).T

    DBNDownBeatTrackingProcessor.process = _patched_dbn_process

    # 加载音频（madmom 自己读文件）
    try:
        sig, sr = load_audio_file(input_path)
    except Exception as e:
        print(f"failed to load audio: {e}", file=sys.stderr)
        sys.exit(4)

    # RNN 下打概率
    rnn = RNNDownBeatProcessor()
    probs = rnn(sig)

    # 下打跟踪：DBN 模型支持 3/4 与 4/4（madmom 0.16 API）
    # fps 必须显式传（RNNDownBeatProcessor 输出帧率 100fps），否则 __init__ 内
    # 60.*fps/max_bpm 会 TypeError
    db_proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    db_result = db_proc(probs)  # [(time, label), ...]  label 1=downbeat, 2..4=beat

    downbeats = [float(t) for t, lab in db_result if lab == 1]
    beats = [float(t) for t, _ in db_result]

    # 估计全局 BPM（RNNDownBeatProcessor 输出 2 列 (beat, downbeat)，
    # TempoEstimationProcessor 取 beat activation 列）
    tempo_proc = TempoEstimationProcessor(fps=100)
    tempi = np.asarray(tempo_proc(probs[:, 0]))
    # tempi 是 (bpm, strength) 的二维数组，取 strength 最高
    if len(tempi):
        best = tempi[np.argmax(tempi[:, 1])]
        bpm = float(best[0])
        strength = float(best[1])
        # v0.5.3 修：strength 本身就是 [0,1] 的概率值（所有 tempo 假设的
        # 直方图归一化强度），原 /100 缩放把它压成 ~0.003 的"恒低置信"谎言，
        # 下游低置信阈值永远触发或永远没人敢信
        tempo_confidence = min(1.0, strength)
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
    # v0.5.1 修 5：wrapper 脚本写系统临时目录，不再泄漏进产物目录
    wrapper_path = Path(tempfile.gettempdir()) / f"mujik_madmom_wrapper_{os.getpid()}.py"
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
        # 尾部截取：traceback 的真实异常在最后几行
        raise MadmomAdapterError(
            f"madmom failed (exit={result.returncode}): {result.stderr[-2000:]}"
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

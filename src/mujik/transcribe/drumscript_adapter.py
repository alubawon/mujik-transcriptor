"""DrumScript 鼓转录 adapter（subprocess 隔离），v0.5.2。

历史：v0.2.x–v0.5.1 用 adtof（MZehren/ADTOF）。2026-09 弃用，原因：
  1. pyproject 里的 git URL（Music-and-Culture-Technology-Lab/Adtof）已 404 死链
  2. 原仓库 LICENSE 实际是 CC-BY-NC-SA 4.0（非商用，违反主线许可证策略；
     docs/research.md 2.3 节当时标的 "MIT" 有误）
  3. 唯一 PyTorch 移植 xavriley/ADTOF-pytorch 无 LICENSE 文件（默认保留所有权利）

替代：DrumScript（Apache-2.0，https://github.com/DrumScript/DrumScript）。
规则引擎（librosa 频谱特征 + 物理规则分类），无模型权重 → 无"权重孤儿"风险；
规则构建素材以技术死亡金属为主，metal blast beat 是明确目标场景。
alpha 阶段已知短板：jazz/funk 准确率一般（官方自述"速度优先于准确率"）。

调用方式：绕过 drumscript.transcribe() 高层封装（它会把事件量化到自身
BPM 检测的 16 分网格并写 PDF），直接用其公开的底层积木：
  load_audio → detect_onsets → classify_events
拿到**未量化**的原始事件，交给主线 rhythm/quantize 层统一处理。

输出格式：subprocess 写 CSV [time_s, instrument, velocity]
velocity 恒为 1.0（DrumScript 不输出逐击力度，其 MIDI 导出也用固定
velocity=100）；本模块按 default_velocity 统一定标。

GM 鼓映射（与 DrumScript DRUM_NOTATION_MAP 一致）：
  kick / kick_clicky → 36 (Bass Drum 1)
  snare              → 38 (Acoustic Snare)
  hi_hat_closed      → 42 (Closed Hi-Hat)
  hi_hat_open        → 46 (Open Hi-Hat)
  low_tom            → 41 (Low Floor Tom)
  mid_tom            → 45 (Low Tom)
  high_tom           → 48 (Hi-Mid Tom)
  crash              → 49 (Crash Cymbal 1)
  ride               → 51 (Ride Cymbal 1)
  unknown            → 跳过（计数告警）
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

from mujik.config.schema import DrumScriptConfig
from mujik.midi.io import DRUM_CHANNEL
from mujik.midi.model import Note

# DrumScript instrument name → GM 标准鼓 note number
GM_DRUM_MAP_DRUMSCRIPT: dict[str, int] = {
    "kick": 36,
    "kick_clicky": 36,
    "snare": 38,
    "hi_hat_closed": 42,
    "hi_hat_open": 46,
    "low_tom": 41,
    "mid_tom": 45,
    "high_tom": 48,
    "crash": 49,
    "ride": 51,
}

# 调用脚本（写进临时文件后 subprocess 执行）
# 直接用底层积木拿原始（未量化）事件；time_sec 为秒，velocity 恒 1.0
_DRUMSCRIPT_WRAPPER = r'''
"""DrumScript 调用 wrapper：参数 <input_audio> <output_csv>"""
import csv
import sys

def main():
    if len(sys.argv) < 3:
        print("usage: _drumscript_wrapper.py <input> <output_csv>", file=sys.stderr)
        sys.exit(2)
    input_path = sys.argv[1]
    output_csv = sys.argv[2]

    try:
        import drumscript
    except ImportError:
        print("drumscript not installed; install via `pip install drumscript`", file=sys.stderr)
        sys.exit(3)

    audio, sr = drumscript.load_audio(input_path)
    onsets = drumscript.detect_onsets(audio, sr)
    events = drumscript.classify_events(audio, sr, onsets)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "instrument", "velocity"])
        for ev in events:
            t = float(ev["time_sec"])
            for inst in ev.get("instruments", []):
                writer.writerow([t, inst, 1.0])

if __name__ == "__main__":
    main()
'''


class DrumScriptAdapterError(RuntimeError):
    pass


def check_drumscript_available() -> bool:
    """检查 drumscript 是否在 venv 中可用。"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import drumscript"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def transcribe_drums_with_drumscript(
    audio_path: str | Path,
    config: DrumScriptConfig | None = None,
    out_dir: str | Path | None = None,
) -> list[Note]:
    """调用 DrumScript 转录鼓 stem 为 Note 列表（固定 channel 9）。

    Args:
        audio_path: 输入音频（应为 demucs 分离出的 drums stem，WAV 最好）
        config: DrumScript 配置
        out_dir: 子进程临时输出目录；None 时用系统 temp

    Returns:
        list[Note]：每个 onset×instrument 一个 Note；
        channel 固定 9，duration = min_note_length_ms / 1000，
        velocity 统一 default_velocity（DrumScript 无逐击力度）
    """
    cfg = config or DrumScriptConfig()
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    if out_dir is None:
        out_dir = Path(tempfile.gettempdir())
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"drumscript_{audio_path.stem}.csv"
    # v0.5.1 修 5 模式：wrapper 脚本写系统临时目录，不泄漏进产物目录
    wrapper_path = Path(tempfile.gettempdir()) / f"mujik_drumscript_wrapper_{os.getpid()}.py"
    wrapper_path.write_text(_DRUMSCRIPT_WRAPPER)

    duration_sec = cfg.min_note_length_ms / 1000.0

    logger.info("drumscript: input={input}", input=audio_path)

    cmd = [sys.executable, str(wrapper_path), str(audio_path), str(csv_path)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=cfg.timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        raise DrumScriptAdapterError(
            f"drumscript timeout after {cfg.timeout_sec}s"
        ) from e

    if result.returncode != 0:
        raise DrumScriptAdapterError(
            f"drumscript failed (exit={result.returncode}): {result.stderr[:500]}"
        )

    if not csv_path.exists():
        raise DrumScriptAdapterError(
            f"drumscript output csv not found: {csv_path}"
        )

    notes: list[Note] = []
    unknown_count = 0
    # 同一物理击打可能被检出多个近邻 onset（量化后才会重合到同一网格点，
    # 原始时间戳仍相邻）→ 同 instrument 在 min_onset_interval_ms 内去重
    min_interval = cfg.min_onset_interval_ms / 1000.0
    last_time_by_pitch: dict[int, float] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row["time_s"])
                instrument = str(row["instrument"])
            except (KeyError, ValueError) as e:
                logger.warning("drumscript: skip malformed row {}: {}", row, e)
                continue

            pitch = GM_DRUM_MAP_DRUMSCRIPT.get(instrument)
            if pitch is None:
                unknown_count += 1
                continue

            last_t = last_time_by_pitch.get(pitch)
            if last_t is not None and (t - last_t) < min_interval:
                continue
            last_time_by_pitch[pitch] = t

            notes.append(Note(
                start=t,
                end=t + duration_sec,
                pitch=pitch,
                velocity=cfg.default_velocity,
                channel=DRUM_CHANNEL,
            ))

    if unknown_count:
        logger.warning("drumscript: {n} unknown-instrument events skipped", n=unknown_count)

    # 清理 wrapper（CSV 留给调用方调试）
    try:
        wrapper_path.unlink()
    except OSError:
        pass

    notes.sort(key=lambda n: n.start)
    logger.info("drumscript: {n} drum events", n=len(notes))
    return notes


__all__ = [
    "transcribe_drums_with_drumscript",
    "check_drumscript_available",
    "DrumScriptAdapterError",
    "GM_DRUM_MAP_DRUMSCRIPT",
]

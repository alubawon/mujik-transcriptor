"""生成 5s 合成测试 wav，用于 E2E smoke test。

构造：
- 440Hz 正弦 0-3s（模拟旋律）
- 200Hz 短脉冲 1.0/2.0/3.0/4.0s（模拟鼓）
- 100Hz 持续 1-4s（模拟贝斯）

用法：python tests/fixtures/generate_synthetic_wav.py [output_path]
默认输出：tests/fixtures/synthetic_5s.wav
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def make_synthetic(duration: float = 5.0, sample_rate: int = 44100) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.zeros_like(t)

    # 旋律：440Hz 0-3s
    mask = t < 3.0
    audio[mask] += 0.2 * np.sin(2 * np.pi * 440 * t[mask])

    # 鼓点：200Hz 短脉冲在 1.0/2.0/3.0/4.0s
    for kick_t in (1.0, 2.0, 3.0, 4.0):
        pulse_mask = (t >= kick_t) & (t < kick_t + 0.1)
        audio[pulse_mask] += 0.3 * np.sin(2 * np.pi * 200 * t[pulse_mask])

    # 贝斯：100Hz 持续 1-4s
    bass_mask = (t >= 1.0) & (t < 4.0)
    audio[bass_mask] += 0.15 * np.sin(2 * np.pi * 100 * t[bass_mask])

    return audio.astype(np.float32)


def main() -> int:
    if len(sys.argv) > 1:
        out_path = Path(sys.argv[1])
    else:
        out_path = Path(__file__).parent / "synthetic_5s.wav"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio = make_synthetic()
    import soundfile as sf
    sf.write(str(out_path), audio, 44100)
    print(f"wrote {out_path} ({audio.shape[0]} samples, {audio.shape[0]/44100:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""响度归一（pyloudnorm，in-proc）。

读取音频 → 测 LUFS → 增益补偿 → 写临时 wav → 返回路径。

回退策略：测得 LUFS 为 -inf（静音/过短）时退回峰值归一。
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
from loguru import logger

from mujik.config.schema import LoudnormConfig

try:
    import pyloudnorm as pyln
    _HAS_PYLOUDNORM = True
except ImportError:
    pyln = None  # type: ignore[assignment]
    _HAS_PYLOUDNORM = False

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    sf = None  # type: ignore[assignment]
    _HAS_SOUNDFILE = False


class LoudnormError(RuntimeError):
    pass


def _check_deps() -> None:
    if not _HAS_SOUNDFILE:
        raise LoudnormError(
            "soundfile not installed; install via "
            "`uv pip install mujik-transcriptor[core-io]`"
        )
    if not _HAS_PYLOUDNORM:
        raise LoudnormError(
            "pyloudnorm not installed; install via "
            "`uv pip install mujik-transcriptor[loudnorm]`"
        )


def _measure_loudness(audio: np.ndarray, sample_rate: int) -> float:
    """测 LUFS；失败返回 -inf。"""
    assert pyln is not None
    meter = pyln.Meter(sample_rate)
    try:
        loudness = meter.integrated_loudness(audio)
    except ValueError as e:
        # 音频太短或全部静音
        logger.debug("loudness measure failed: {}", e)
        return float("-inf")
    return float(loudness)


def _peak_normalize(audio: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    """峰值归一到 target_dbfs dBFS。"""
    peak = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if peak <= 0.0:
        return audio
    target_amp = 10 ** (target_dbfs / 20.0)
    gain = target_amp / peak
    return (audio * gain).astype(audio.dtype, copy=False)


def normalize_loudness(
    audio_path: str | Path,
    config: LoudnormConfig | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """把音频响度归一到目标 LUFS。

    Args:
        audio_path: 输入音频路径（WAV/FLAC/AIFF 等 soundfile 支持的格式）
        config: 响度归一配置；None 时用默认（target_lufs=-14, peak_dbfs=-1）
        out_path: 输出路径；None 时写到 tempfile.NamedTemporaryFile

    Returns:
        归一化后的 wav 文件路径

    Raises:
        FileNotFoundError: 输入不存在
        LoudnormError: 依赖缺失 / 写入失败
    """
    _check_deps()

    cfg = config or LoudnormConfig()
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    if out_path is None:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", prefix="mujik_loudnorm_", delete=False
        )
        tmp.close()
        out_path = Path(tmp.name)
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    assert sf is not None
    audio, sample_rate = sf.read(str(audio_path), always_2d=False)
    original_dtype = audio.dtype
    logger.info(
        "loudnorm: input={input}, sr={sr}, shape={shape}, dtype={dtype}",
        input=audio_path, sr=sample_rate,
        shape=audio.shape, dtype=original_dtype,
    )

    # 转 float64 给 pyloudnorm（内部用 K-weighting 滤波）
    if audio.dtype != np.float64:
        audio_f = audio.astype(np.float64)
    else:
        audio_f = audio

    loudness = _measure_loudness(audio_f, sample_rate)
    if math.isinf(loudness):
        logger.warning(
            "loudnorm: input is silent or too short (LUFS=-inf), "
            "falling back to peak normalization (target={dbfs} dBFS)",
            dbfs=cfg.peak_dbfs,
        )
        normalized = _peak_normalize(audio_f, target_dbfs=cfg.peak_dbfs)
        used_method = "peak"
    else:
        assert pyln is not None
        normalized = pyln.normalize.loudness(
            audio_f, loudness, float(cfg.target_lufs)
        )
        # 防止 pyloudnorm 输出超过 0 dBFS 削波
        peak = float(np.max(np.abs(normalized))) if normalized.size > 0 else 0.0
        if peak > 1.0:
            logger.warning(
                "loudnorm: output would clip (peak={peak:.3f}), scaling down",
                peak=peak,
            )
            normalized = normalized / peak * 0.99
        used_method = f"lufs_{cfg.target_lufs}"

    # 转回原 dtype
    if original_dtype != np.float64:
        normalized = normalized.astype(original_dtype)

    assert sf is not None
    sf.write(str(out_path), normalized, sample_rate)
    logger.info(
        "loudnorm: done ({method}) → {out}, sr={sr}, peak={peak:.3f}",
        method=used_method, out=out_path, sr=sample_rate,
        peak=float(np.max(np.abs(normalized))) if normalized.size > 0 else 0.0,
    )
    return out_path


__all__ = [
    "normalize_loudness",
    "LoudnormError",
]

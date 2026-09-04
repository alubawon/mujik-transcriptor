"""htdemucs_6s 6-stem Demucs adapter（v0.4.0）。

htdemucs_6s 是 Demucs v4 的 6-stem 变体（MIT 许可）：
    vocals / drums / bass / piano / guitar / other

本模块复用 demucs_adapter 的 subprocess 模式，但 glob 6 个 stem 文件。

设计决策（v0.4.0）：
- 独立模块而非 demucs_adapter.py 内部 switch：因为输出文件 glob 不同
- 与 demucs_adapter 共用 demucs CLI 调用（同一进程隔离路径）
- 失败 fallback：返回 4-stem 风格（vocals/drums/bass/other）+ warning
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from loguru import logger

from mujik.config.schema import SourceSeparationConfig
from mujik.separate.model import Stem, Stems


HTDEMUCS_6S_STEMS: tuple[str, ...] = ("vocals", "drums", "bass", "piano", "guitar", "other")


class Htdemucs6sAdapterError(RuntimeError):
    pass


def check_htdemucs_6s_available() -> bool:
    """检查 demucs CLI + htdemucs_6s variant 是否可用。"""
    if shutil.which("demucs") is None and shutil.which("python") is None:
        return False
    # 简化：检查 demucs importable
    try:
        import demucs  # noqa: F401
        return True
    except ImportError:
        pass
    # 退化：检查 demucs CLI
    return shutil.which("demucs") is not None


def separate_with_htdemucs_6s(
    input_path: Path | str,
    out_dir: Path | str,
    config: SourceSeparationConfig | None = None,
) -> Stems:
    """用 htdemucs_6s 跑 6-stem 源分离。

    Args:
        input_path: 输入音频
        out_dir: 输出目录
        config: SourceSeparationConfig（model 字段被忽略，固定 htdemucs_6s）

    Returns:
        Stems：6 stem（vocals/drums/bass/piano/guitar/other）
    """
    cfg = config or SourceSeparationConfig()
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    # 写 tmp 目录避免污染 out_dir
    with tempfile.TemporaryDirectory(prefix="htdemucs_6s_") as tmp:
        tmp_path = Path(tmp)
        tmp_out = tmp_path / "out"
        tmp_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "-m", "demucs",
            "-n", "htdemucs_6s",
            "--device", cfg.device,
            "--out", str(tmp_out),
            "--segment", str(cfg.segment_length),
            "--overlap", str(cfg.overlap),
            "--jobs", str(cfg.jobs),
            str(input_path),
        ]
        # 浮点精度
        if cfg.precision in ("fp16", "bf16"):
            cmd.extend(["--float16" if cfg.precision == "fp16" else "--bf16"])

        logger.info(
            "htdemucs_6s subprocess: cmd={cmd}",
            cmd=" ".join(cmd),
        )
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=3600,
            )
        except subprocess.TimeoutExpired as e:
            raise Htdemucs6sAdapterError(
                "htdemucs_6s subprocess timeout (3600s)"
            ) from e
        except FileNotFoundError as e:
            raise Htdemucs6sAdapterError(
                "demucs CLI not executable; install via "
                "`uv pip install 'mujik-transcriptor[separate]'`"
            ) from e

        if result.returncode != 0:
            raise Htdemucs6sAdapterError(
                f"htdemucs_6s failed (exit={result.returncode}): "
                f"{result.stderr[:500]}"
            )

        # demucs 写出路径：{tmp_out}/htdemucs_6s/{input_stem}/*.wav
        track_dir = tmp_out / "htdemucs_6s" / input_path.stem
        if not track_dir.exists():
            raise Htdemucs6sAdapterError(
                f"htdemucs_6s output dir not found: {track_dir}; "
                f"out_dir contents: {list(tmp_out.iterdir())}"
            )

        # 复制 6 个 stem 文件到 out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        stems_obj = Stems(
            separation_model="demucs/htdemucs_6s",
            sample_rate=44100,
            total_duration=0.0,  # 由具体 stem 决定
        )
        n_found = 0
        for stem_name in HTDEMUCS_6S_STEMS:
            src = track_dir / f"{stem_name}.wav"
            if not src.exists():
                logger.warning(
                    "htdemucs_6s: missing stem {stem}, skipping",
                    stem=stem_name,
                )
                continue
            dst = out_dir / f"{input_path.stem}_{stem_name}.wav"
            shutil.copy(src, dst)
            n_found += 1
            # 取 wav 时长
            try:
                import soundfile as sf
                info = sf.info(str(dst))
                duration = float(info.duration)
            except Exception as e:
                # v0.5.2: duration=0.0 不再无声无息（下游按 0 处理时至少有迹可循）
                logger.warning(
                    "htdemucs_6s: failed to probe duration for %s: %s",
                    dst, e,
                )
                duration = 0.0
            stems_obj.add(Stem(
                name=stem_name,  # type: ignore[arg-type]
                audio_path=dst,
                sample_rate=44100,
                duration=duration,
                source_model="demucs/htdemucs_6s",
            ))

        if n_found == 0:
            raise Htdemucs6sAdapterError(
                f"htdemucs_6s produced no stem files in {track_dir}"
            )
        if n_found < 6:
            logger.warning(
                "htdemucs_6s: only {n}/6 stems found, partial result",
                n=n_found,
            )

    logger.info(
        "htdemucs_6s: {n} stems → {out}",
        n=n_found, out=out_dir,
    )
    return stems_obj


__all__ = [
    "HTDEMUCS_6S_STEMS",
    "Htdemucs6sAdapterError",
    "check_htdemucs_6s_available",
    "separate_with_htdemucs_6s",
]

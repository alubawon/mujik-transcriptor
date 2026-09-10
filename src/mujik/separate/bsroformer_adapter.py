"""BS-Roformer SW 6-stem 分离 adapter（v0.5.3）。

BS-Roformer 是 ByteDance 提出的频带 Roformer 架构（arXiv:2309.02612，
ICASSP 2024，SDX23 冠军）。**ByteDance 只发论文，没有官方代码/权重**，
生态全靠第三方：lucidrains 的 MIT 复现（不含权重）+ 社区训练的 checkpoint。

本 adapter 走 `bs-roformer-infer` CLI（MIT），默认 registry 模型
`roformer-model-bs-roformer-sw-by-jarredou` —— 目前**唯一**产出独立
guitar/piano 两轨的 Roformer checkpoint。

为什么 subprocess 而非 in-process（与 madmom/BTC 同一先例）：
- 依赖栈重且与主线冲突（自带 torch/rotary-embedding-torch/einops 版本约束）
- 权重来源不可信（见下），进程隔离便于随时摘掉

⚠️ **权重许可警告（重要，不要静默）**：
    代码是 MIT，但**权重许可未声明**。原作者 jarredou 的 HF 账号已于
    2026-06 注销，现存镜像 `enerjazzer/BS-ROFO-SW-Fixed` 页面显示
    "License: unknown"。sha256 能证明字节一致，**无法证明任何人有分发权**。
    因此本 adapter：
      1. 默认**不启用**（router 里要显式配 model=bsroformer）
      2. 每次调用都打 license warning（config 不说谎，许可不装懂）
      3. 仅推荐用于**本地评测/研究**；主线商业分发前必须解决许可问题
    参考：ZFTurbo/MSST issue #245/#248/#249（权重许可追问，至今零回复）

注：我们现用的 htdemucs 权重亦非完全干净（facebookresearch/demucs#327：
Meta 预训练权重"仅供科研用途"），这不是本模型独有的问题。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from loguru import logger

from mujik.config.schema import SourceSeparationConfig
from mujik.separate.model import Stem, Stems

# BS-Roformer SW 产出 6 stem（+ 一个 instrumental，非 StemName，不纳入 Stems）
BSROFORMER_STEMS: tuple[str, ...] = (
    "vocals", "drums", "bass", "piano", "guitar", "other",
)

# 默认 registry slug（bs-roformer-infer 的 --model）
DEFAULT_BSROFORMER_MODEL = "roformer-model-bs-roformer-sw-by-jarredou"

# CLI 可执行文件的环境变量覆盖（因权重/依赖隔离，通常装在独立 venv 里）
BSROFORMER_BIN_ENV = "MUJIK_BSROFORMER_BIN"

_LICENSE_WARNING = (
    "BS-Roformer SW 权重许可未声明（原作者 HF 账号已注销，镜像标 "
    "'License: unknown'）。代码 MIT，但权重**不可假定可分发**——"
    "仅用于本地评测/研究，商业分发前须解决许可。见本模块 docstring。"
)


class BsRoformerAdapterError(RuntimeError):
    """bs-roformer-infer CLI 不可用或执行失败。"""


def _resolve_bin() -> str:
    """定位 bs-roformer-infer 可执行文件。

    优先 $MUJIK_BSROFORMER_BIN（独立 venv 的常见情形），否则查 PATH。
    """
    override = os.environ.get(BSROFORMER_BIN_ENV)
    if override:
        if not Path(override).exists():
            raise BsRoformerAdapterError(
                f"${BSROFORMER_BIN_ENV}={override} 指向的文件不存在"
            )
        return override
    found = shutil.which("bs-roformer-infer")
    if found is None:
        raise BsRoformerAdapterError(
            "bs-roformer-infer 不在 PATH 中。因依赖栈与主线冲突，建议装到独立 "
            "venv 并用 $MUJIK_BSROFORMER_BIN 指向其 bin："
            "`uv venv bsrofo-venv && uv pip install --python "
            "./bsrofo-venv/bin/python bs-roformer-infer`"
        )
    return found


def check_bsroformer_available() -> bool:
    """检查 bs-roformer-infer CLI 是否可用（不下载权重）。"""
    try:
        _resolve_bin()
        return True
    except BsRoformerAdapterError:
        return False


def separate_with_bsroformer(
    input_path: str | Path,
    out_dir: str | Path,
    config: SourceSeparationConfig | None = None,
) -> Stems:
    """用 BS-Roformer SW 跑 6-stem 源分离（含独立 guitar/piano）。

    Args:
        input_path: 输入音频
        out_dir: 输出目录
        config: 分离配置；variant 可写 registry slug（默认 SW 6-stem 模型）

    Returns:
        Stems：6 stem（vocals/drums/bass/piano/guitar/other）

    Raises:
        BsRoformerAdapterError: CLI 缺失/执行失败/无产物
        FileNotFoundError: 输入文件不存在
    """
    import soundfile as sf

    cfg = config or SourceSeparationConfig()
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    binary = _resolve_bin()
    # 权重许可每次都说，不因跑过就闭嘴
    logger.warning(_LICENSE_WARNING)

    # variant 默认值是 demucs 的，别拿去当 registry slug
    model_slug = (
        cfg.variant
        if cfg.variant and cfg.variant.startswith("roformer-")
        else DEFAULT_BSROFORMER_MODEL
    )

    if cfg.precision in ("fp16", "bf16"):
        # CLI 内部对 MPS/CUDA 自行选精度，无外部 flag——不假装我们控制了它
        logger.warning(
            "bs-roformer-infer 无精度 flag（由 CLI 内部按 device 决定）；"
            "cfg.precision={p} 未生效", p=cfg.precision,
        )

    # CLI 是"目录进、目录出"，且按输入文件 stem 命名产物 → 用独立 tmp 输入目录，
    # 避免把同目录其它音频一起卷进去分离
    with tempfile.TemporaryDirectory(prefix="bsroformer_") as tmp:
        tmp_path = Path(tmp)
        tmp_in = tmp_path / "in"
        tmp_out = tmp_path / "out"
        tmp_in.mkdir(parents=True, exist_ok=True)
        tmp_out.mkdir(parents=True, exist_ok=True)

        # 软链省一次大文件拷贝；失败（跨卷等）退化为 copy
        staged = tmp_in / input_path.name
        try:
            staged.symlink_to(input_path.resolve())
        except OSError:
            shutil.copy(input_path, staged)

        cmd = [
            binary,
            "--model", model_slug,
            "--input_folder", str(tmp_in),
            "--store_dir", str(tmp_out),
            "--device", cfg.device,
        ]
        logger.info("bs-roformer subprocess: cmd={cmd}", cmd=" ".join(cmd))

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200,
            )
        except subprocess.TimeoutExpired as e:
            raise BsRoformerAdapterError(
                "bs-roformer-infer timeout (7200s)"
            ) from e
        except FileNotFoundError as e:
            raise BsRoformerAdapterError(
                f"bs-roformer-infer 不可执行: {binary}"
            ) from e

        if result.returncode != 0:
            # 真实异常在 stderr 尾部，头部多是 torch FutureWarning
            raise BsRoformerAdapterError(
                f"bs-roformer-infer failed (exit={result.returncode}): "
                f"{result.stderr[-2000:]}"
            )

        # 产物是平铺的 {input_stem}_{stem}.wav（另有 _instrumental.wav，不收）
        out_dir.mkdir(parents=True, exist_ok=True)
        stems_obj = Stems(
            separation_model=f"bsroformer/{model_slug}",
            sample_rate=44100,
            total_duration=0.0,
        )
        n_found = 0
        sample_rate = 44100
        max_duration = 0.0
        for stem_name in BSROFORMER_STEMS:
            src = tmp_out / f"{input_path.stem}_{stem_name}.wav"
            if not src.exists():
                logger.warning(
                    "bs-roformer: missing stem {stem}, skipping", stem=stem_name,
                )
                continue
            dst = out_dir / f"{input_path.stem}_{stem_name}.wav"
            shutil.move(str(src), dst)
            n_found += 1
            try:
                info = sf.info(str(dst))
                duration = float(info.duration)
                sample_rate = info.samplerate
                max_duration = max(max_duration, duration)
            except Exception as e:  # noqa: BLE001
                # duration=0.0 不无声无息（下游按 0 处理时至少有迹可循）
                logger.warning(
                    "bs-roformer: failed to probe duration for {dst}: {e}",
                    dst=dst, e=e,
                )
                duration = 0.0
            stems_obj.add(Stem(
                name=stem_name,  # type: ignore[arg-type]
                audio_path=dst,
                sample_rate=sample_rate,
                duration=duration,
                source_model=f"bsroformer/{model_slug}",
                metadata={"weights_license": "unknown"},
            ))

        if n_found == 0:
            raise BsRoformerAdapterError(
                f"bs-roformer-infer produced no stem files in {tmp_out}; "
                f"contents: {list(tmp_out.iterdir())}"
            )
        if n_found < len(BSROFORMER_STEMS):
            logger.warning(
                "bs-roformer: only {n}/{total} stems found, partial result",
                n=n_found, total=len(BSROFORMER_STEMS),
            )

    stems_obj.sample_rate = sample_rate
    stems_obj.total_duration = max_duration
    logger.info("bs-roformer: {n} stems → {out}", n=n_found, out=out_dir)
    return stems_obj


__all__ = [
    "BSROFORMER_STEMS",
    "BSROFORMER_BIN_ENV",
    "DEFAULT_BSROFORMER_MODEL",
    "BsRoformerAdapterError",
    "check_bsroformer_available",
    "separate_with_bsroformer",
]

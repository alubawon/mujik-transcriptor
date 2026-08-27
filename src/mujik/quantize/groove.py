"""Quantize groove 模板（v0.2.3 minimal: straight + swing16）。

职责：给定一个 snap 后的时间点，根据 groove 模板返回额外的偏移（拍数）。

约定：
- groove_offset 返回一个浮点偏移（拍数），caller 用 bpm 转秒
- 模板名只在 straight / swing16 范围内合法；其他抛 ValueError
- swing16：8 分音符的 offbeat 位置按 ratio 后移（默认 ratio=0.6，即 60% 偏 long）
"""
from __future__ import annotations

# 默认 swing ratio：0.5 = 直拍，>0.5 = 偏 swing
DEFAULT_SWING_RATIO: float = 0.6

_VALID_TEMPLATES: tuple[str, ...] = ("straight", "swing16")


def is_offbeat_position(grid_position: int, grid_resolution: int) -> bool:
    """判断 grid_position 是否是 8 分 offbeat（grid idx = grid_resolution / 2）。"""
    return grid_position == grid_resolution // 2


def is_16th_off_offbeat(grid_position: int, grid_resolution: int) -> bool:
    """判断 grid_position 是否在 8 分 offbeat ± 1 的 16 分细分位置。"""
    if grid_resolution < 16:
        return False
    offbeat = grid_resolution // 2
    return abs(grid_position - offbeat) == 1


def groove_offset(
    beat_position: float,
    grid_position: int,
    grid_resolution: int,
    template: str,
    ratio: float = DEFAULT_SWING_RATIO,
) -> float:
    """计算 groove 模板引入的额外偏移（拍数）。

    返回偏移**拍数**（不是秒），caller 用 `bpm` 转秒。
    正值 = 后移，负值 = 前移。
    """
    if template not in _VALID_TEMPLATES:
        raise ValueError(
            f"unknown groove template: {template!r}; valid: {_VALID_TEMPLATES}"
        )
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0,1], got {ratio}")
    if grid_resolution <= 0:
        raise ValueError(f"grid_resolution must be > 0, got {grid_resolution}")

    if template == "straight":
        return 0.0

    # swing16
    if is_offbeat_position(grid_position, grid_resolution):
        return (ratio - 0.5)

    if is_16th_off_offbeat(grid_position, grid_resolution):
        return (ratio - 0.5) * 0.5

    return 0.0


def groove_offset_seconds(
    beat_position: float,
    grid_position: int,
    grid_resolution: int,
    template: str,
    bpm: float,
    ratio: float = DEFAULT_SWING_RATIO,
) -> float:
    """groove_offset 的便捷版本，直接返回秒。"""
    if bpm <= 0:
        raise ValueError(f"bpm must be > 0, got {bpm}")
    beat_dur = 60.0 / bpm
    return groove_offset(beat_position, grid_position, grid_resolution, template, ratio) * beat_dur


__all__ = [
    "DEFAULT_SWING_RATIO",
    "is_offbeat_position",
    "is_16th_off_offbeat",
    "groove_offset",
    "groove_offset_seconds",
]

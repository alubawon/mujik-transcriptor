"""Density filter：当某时刻同时发声 note 数 > max_simultaneous 时，丢弃 velocity 最低的。

算法（v0.2.3）：
  1. 按 start_time 排序
  2. 维护 active note 的最小堆
  3. 处理每个 note：
     a. 弹出已结束的
     b. 若 active 数 + 1 > max → 找 active 中 velocity 最低的 note，从 kept 移除
     c. 新 note 加入 kept + active

复杂度：O(n log n)。
"""
from __future__ import annotations

import heapq
from typing import Optional

from mujik.midi.model import Note


class _Active:
    """active note 包装：可比较（按 end_time → velocity → id）。"""

    __slots__ = ("end_time", "velocity", "kept_index")

    def __init__(self, end_time: float, velocity: int, kept_index: int) -> None:
        self.end_time = end_time
        self.velocity = velocity
        self.kept_index = kept_index

    def __lt__(self, other: "_Active") -> bool:
        # min-heap：end_time 小的先弹
        if self.end_time != other.end_time:
            return self.end_time < other.end_time
        if self.velocity != other.velocity:
            return self.velocity < other.velocity
        return self.kept_index < other.kept_index


def apply_density_filter(
    notes: list[Note],
    max_simultaneous: int,
) -> tuple[list[Note], int]:
    """应用密度过滤。

    Args:
        notes: 输入 note 列表（任意顺序）
        max_simultaneous: 同时发声上限

    Returns:
        (kept_notes 按输入顺序, dropped_count)
    """
    if max_simultaneous <= 0:
        raise ValueError(f"max_simultaneous must be > 0, got {max_simultaneous}")
    if not notes:
        return [], 0

    # 按 start_time 排序，保留原始 index
    sorted_pairs = sorted(enumerate(notes), key=lambda x: (x[1].start, x[0]))
    kept: list[Optional[Note]] = []  # 用 None 标记"被丢弃的槽位"
    active: list[_Active] = []
    dropped = 0
    next_kept_idx = 0  # kept 槽位索引

    for _orig_idx, note in sorted_pairs:
        # 1. 弹掉已结束的
        while active and active[0].end_time <= note.start:
            heapq.heappop(active)

        # 2. 检查是否超限（即将加入的 note 会成为 active）
        if len(active) >= max_simultaneous:
            # active 中必有被淘汰的：找 velocity 最低
            min_idx = min(range(len(active)), key=lambda i: active[i].velocity)
            victim = active[min_idx]
            # 从 kept 中移除 victim
            kept[victim.kept_index] = None
            dropped += 1
            # 从 active 中移除
            active.pop(min_idx)
            heapq.heapify(active)

        # 3. 加入 kept 和 active
        kept.append(note)
        heapq.heappush(active, _Active(note.end, note.velocity, next_kept_idx))
        next_kept_idx += 1

    # 过滤掉被淘汰的槽位，保持原输入顺序
    result: list[Note] = [n for n in kept if n is not None]
    return result, dropped


__all__ = ["apply_density_filter"]

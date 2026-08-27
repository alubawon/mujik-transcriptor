"""Piano reduction：把多声部 / 复调内容压缩为单声部钢琴可读的总谱。

启发式（v0.2.3，简单版本）：
  1. 按 start_time 排序，分组到 onset cluster（同 start 视为同一拍点）
  2. 在同一 onset 上保留 top K by velocity（K = max(1, max_simultaneous // 2)）
  3. Held note（已发声 + 在新 onset 仍 active）：仅当 velocity 高于 active 中位时才保留

更复杂版本（v0.4+）：
  - 音乐理论 aware（chord-tone 优先，voice-leading）
  - 高密度节拍下的 smart drop
"""
from __future__ import annotations

from mujik.midi.model import Note


def _cluster_onsets(notes: list[Note], tolerance: float = 1e-4) -> list[list[Note]]:
    """把 note 按 start_time 分组（容差 tolerance 秒）。

    返回 onset 列表，每个 onset 是一组 note。
    """
    if not notes:
        return []
    sorted_notes = sorted(notes, key=lambda n: (n.start, -n.velocity))
    clusters: list[list[Note]] = [[sorted_notes[0]]]
    for note in sorted_notes[1:]:
        if abs(note.start - clusters[-1][0].start) <= tolerance:
            clusters[-1].append(note)
        else:
            clusters.append([note])
    return clusters


def piano_reduce(
    notes: list[Note],
    max_simultaneous: int,
) -> tuple[list[Note], int]:
    """钢琴缩减。

    Args:
        notes: 输入 notes
        max_simultaneous: 限制同时发声数；同时也是 K = max(1, max_simultaneous // 2) 的来源

    Returns:
        (reduced_notes, dropped_count)

    算法：
      1. 按 start_time 聚类成 onset 簇
      2. 每个 onset 保留 top K by velocity
      3. 持续音 (end - start > 0.1s)：在下一 onset 出现时
         - 仅当其 velocity > 当前 active 中位 velocity 时保留
         - 否则视为"被覆盖"，丢弃
    """
    if max_simultaneous <= 0:
        raise ValueError(f"max_simultaneous must be > 0, got {max_simultaneous}")
    if not notes:
        return [], 0

    k = max(1, max_simultaneous // 2)
    clusters = _cluster_onsets(notes)
    kept: list[Note] = []
    dropped = 0
    # active_held 跟踪当前还"在响"的持续音
    active_held: list[Note] = []

    for cluster in clusters:
        # 1. 处理持续音：用严格 > 比较，只保留高于中位的
        if active_held:
            median_vel = sorted([n.velocity for n in active_held])[len(active_held) // 2]
            new_held: list[Note] = []
            for h in active_held:
                if h.velocity > median_vel:
                    kept.append(h)
                    new_held.append(h)
                else:
                    dropped += 1
            active_held = new_held

        # 2. 当前 onset 上保留 top K by velocity
        cluster_sorted = sorted(cluster, key=lambda n: -n.velocity)
        top_k = cluster_sorted[:k]
        rest = cluster_sorted[k:]

        for n in top_k:
            kept.append(n)
            if n.end - n.start > 0.1:
                active_held.append(n)
        dropped += len(rest)

    return kept, dropped


__all__ = ["piano_reduce"]

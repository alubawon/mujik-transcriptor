"""Tests for ChordEvent.__post_init__ hardening (v0.4.6)."""
from __future__ import annotations

import pytest

from mujik.midi.model import (
    ALLOWED_QUALITIES_BY_VOCAB,
    ChordEvent,
)


class TestRootValidation:
    """v0.4.6: root 必须 ^[A-Ga-g][#b]?$。"""

    def test_valid_root_c(self):
        c = ChordEvent(0.0, 1.0, "C", "")
        assert c.root == "C"

    def test_valid_root_sharp(self):
        c = ChordEvent(0.0, 1.0, "F#", "")
        assert c.root == "F#"

    def test_valid_root_flat(self):
        c = ChordEvent(0.0, 1.0, "Bb", "")
        assert c.root == "Bb"

    def test_lowercase_root_accepted(self):
        """v0.4.6: 大小写不敏感。"""
        c = ChordEvent(0.0, 1.0, "c", "")
        assert c.root == "c"

    def test_empty_root_rejected(self):
        with pytest.raises(ValueError, match="root must match"):
            ChordEvent(0.0, 1.0, "", "")

    def test_h_rejected(self):
        """v0.4.6: H 不是合法 root（德式记号 B）。"""
        with pytest.raises(ValueError, match="root must match"):
            ChordEvent(0.0, 1.0, "H", "")

    def test_double_sharp_rejected(self):
        with pytest.raises(ValueError, match="root must match"):
            ChordEvent(0.0, 1.0, "C##", "")

    def test_multi_letter_rejected(self):
        with pytest.raises(ValueError, match="root must match"):
            ChordEvent(0.0, 1.0, "Do", "")

    def test_digit_rejected(self):
        with pytest.raises(ValueError, match="root must match"):
            ChordEvent(0.0, 1.0, "C1", "")


class TestBassValidation:
    """v0.4.6: bass 同 root 规则；空字符串允许。"""

    def test_empty_bass_allowed(self):
        c = ChordEvent(0.0, 1.0, "C", "", bass="")
        assert c.bass == ""

    def test_valid_bass(self):
        c = ChordEvent(0.0, 1.0, "C", "7", bass="Bb")
        assert c.bass == "Bb"

    def test_invalid_bass_rejected(self):
        with pytest.raises(ValueError, match="bass must match"):
            ChordEvent(0.0, 1.0, "C", "7", bass="H")


class TestStartEnd:
    """v0.4.6: start >= 0, end >= start。"""

    def test_equal_start_end_allowed(self):
        """v0.4.6: 允许 placeholder（madmom_adapter 内部使用）。"""
        c = ChordEvent(0.0, 0.0, "C", "")
        assert c.end == c.start

    def test_negative_start_rejected(self):
        with pytest.raises(ValueError, match="start must be >= 0"):
            ChordEvent(-0.1, 1.0, "C", "")

    def test_end_before_start_rejected(self):
        with pytest.raises(ValueError, match="end"):
            ChordEvent(2.0, 1.0, "C", "")


class TestQualityValidation:
    """v0.4.6: quality 由 vocab 决定允许集合。"""

    def test_root_vocab_only_empty(self):
        """v0.4.6: 'root' vocab 只接受空 quality。"""
        # 空 quality 允许
        c = ChordEvent(0.0, 1.0, "C", "", vocab="root")
        assert c.quality == ""
        # 非空 quality 拒绝
        with pytest.raises(ValueError, match="quality"):
            ChordEvent(0.0, 1.0, "C", "m", vocab="root")

    def test_root_quality_vocab(self):
        """v0.4.6: 'root-quality' vocab 接受 maj/min/m。"""
        for q in ("", "maj", "major", "M", "m", "min", "minor", "-"):
            c = ChordEvent(0.0, 1.0, "C", q, vocab="root-quality")
            assert c.quality == q
        # 7/maj7 在 root-quality vocab 内被拒
        with pytest.raises(ValueError, match="quality"):
            ChordEvent(0.0, 1.0, "C", "7", vocab="root-quality")

    def test_extended_vocab_default(self):
        """v0.4.6: 'extended' vocab 接受 7/maj7/m7/dim/aug/sus。"""
        for q in ("", "m", "7", "maj7", "m7", "dim", "aug", "sus", "sus4"):
            c = ChordEvent(0.0, 1.0, "C", q)  # 默认 vocab=extended
            assert c.quality == q

    def test_extended_vocab_rejects_9(self):
        """v0.4.6: 9/11/13/alt 在 extended 外（v0.4.8 BTC-HCQT 评估）。"""
        with pytest.raises(ValueError, match="quality"):
            ChordEvent(0.0, 1.0, "C", "9")

    def test_extended_vocab_rejects_alt(self):
        with pytest.raises(ValueError, match="quality"):
            ChordEvent(0.0, 1.0, "C", "alt")

    def test_unknown_vocab_rejected(self):
        with pytest.raises(ValueError, match="vocab"):
            ChordEvent(0.0, 1.0, "C", "", vocab="super-extended")

    def test_allowed_qualities_dict(self):
        """v0.4.6: ALLOWED_QUALITIES_BY_VOCAB 三档存在。"""
        assert "root" in ALLOWED_QUALITIES_BY_VOCAB
        assert "root-quality" in ALLOWED_QUALITIES_BY_VOCAB
        assert "extended" in ALLOWED_QUALITIES_BY_VOCAB
        # 集合是 frozenset
        for s in ALLOWED_QUALITIES_BY_VOCAB.values():
            assert isinstance(s, frozenset)
        # root ⊂ root-quality ⊂ extended
        assert ALLOWED_QUALITIES_BY_VOCAB["root"].issubset(
            ALLOWED_QUALITIES_BY_VOCAB["root-quality"]
        )
        assert ALLOWED_QUALITIES_BY_VOCAB["root-quality"].issubset(
            ALLOWED_QUALITIES_BY_VOCAB["extended"]
        )


class TestValidConstruction:
    """v0.4.6: 合法 case 顺利构造。"""

    def test_minimal_chord(self):
        c = ChordEvent(0.0, 1.0, "C")
        assert c.start == 0.0
        assert c.end == 1.0
        assert c.root == "C"
        assert c.quality == ""
        assert c.bass == ""
        assert c.vocab == "extended"

    def test_full_chord(self):
        c = ChordEvent(0.0, 1.0, "F#", "m7", bass="A", vocab="extended")
        assert c.root == "F#"
        assert c.quality == "m7"
        assert c.bass == "A"

    def test_slash_chord(self):
        c = ChordEvent(0.0, 1.0, "C", "7", bass="Bb")
        assert c.bass == "Bb"

    def test_dataclass_equality(self):
        """v0.4.6: 相同字段 → 相等（frozen 行为）。"""
        c1 = ChordEvent(0.0, 1.0, "C", "")
        c2 = ChordEvent(0.0, 1.0, "C", "")
        assert c1 == c2

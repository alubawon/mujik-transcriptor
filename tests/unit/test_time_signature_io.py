"""Tests for time_signature/io.py round-trip."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mujik.time_signature.io import (
    read_time_signatures_json,
    write_time_signatures_json,
)
from mujik.time_signature.model import TimeSignatureSegment


def _seg(start: float, end: float, sig=(4, 4), conf=1.0, source="manual") -> TimeSignatureSegment:
    return TimeSignatureSegment(
        start_time=start,
        end_time=end,
        time_signature=sig,
        confidence=conf,
        source=source,  # type: ignore[arg-type]
    )


class TestRoundTrip:
    def test_single_segment(self, tmp_path: Path):
        segs = [_seg(0.0, 10.0, (4, 4), 0.9, "default_4_4")]
        p = tmp_path / "ts.json"
        write_time_signatures_json(segs, p)
        loaded = read_time_signatures_json(p)
        assert len(loaded) == 1
        assert loaded[0].start_time == 0.0
        assert loaded[0].end_time == 10.0
        assert loaded[0].time_signature == (4, 4)
        assert loaded[0].confidence == 0.9
        assert loaded[0].source == "default_4_4"

    def test_multiple_segments(self, tmp_path: Path):
        segs = [
            _seg(0.0, 5.0, (4, 4), 0.8, "auto_beatnet"),
            _seg(5.0, 12.0, (3, 4), 0.95, "manual"),
            _seg(12.0, 20.0, (7, 8), 0.7, "manual"),
        ]
        p = tmp_path / "ts.json"
        write_time_signatures_json(segs, p)
        loaded = read_time_signatures_json(p)
        assert len(loaded) == 3
        assert loaded[0].time_signature == (4, 4)
        assert loaded[1].time_signature == (3, 4)
        assert loaded[2].time_signature == (7, 8)
        assert loaded[1].source == "manual"

    def test_compatible_with_pipeline_format(self, tmp_path: Path):
        """匹配 v0.2.2 pipeline 写出的格式：start/end/sig 键名。"""
        raw = [
            {"start": 0.0, "end": 10.0, "sig": [4, 4], "confidence": 0.3, "source": "default_4_4"},
        ]
        p = tmp_path / "ts.json"
        p.write_text(json.dumps(raw))
        loaded = read_time_signatures_json(p)
        assert loaded[0].time_signature == (4, 4)
        assert loaded[0].source == "default_4_4"

    def test_compatible_with_long_key_names(self, tmp_path: Path):
        """也接受 start_time/end_time/time_signature 完整命名。"""
        raw = [
            {"start_time": 0.0, "end_time": 8.0, "time_signature": [3, 4], "confidence": 0.5, "source": "manual"},
        ]
        p = tmp_path / "ts.json"
        p.write_text(json.dumps(raw))
        loaded = read_time_signatures_json(p)
        assert loaded[0].time_signature == (3, 4)

    def test_missing_file_returns_empty(self, tmp_path: Path):
        p = tmp_path / "nope.json"
        assert read_time_signatures_json(p) == []


class TestAtomicWrite:
    def test_no_partial_file(self, tmp_path: Path):
        """写盘后不应残留 .tmp。"""
        segs = [_seg(0.0, 5.0)]
        p = tmp_path / "ts.json"
        write_time_signatures_json(segs, p)
        assert p.exists()
        assert not (p.with_suffix(p.suffix + ".tmp")).exists()

    def test_creates_parent_dir(self, tmp_path: Path):
        segs = [_seg(0.0, 5.0)]
        p = tmp_path / "deep" / "nest" / "ts.json"
        write_time_signatures_json(segs, p)
        assert p.exists()


class TestEdgeCases:
    def test_invalid_sig_raises(self, tmp_path: Path):
        raw = [{"start": 0.0, "end": 1.0, "sig": [4, 3], "confidence": 0.5, "source": "manual"}]
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(raw))
        with pytest.raises(ValueError):
            read_time_signatures_json(p)

    def test_unknown_source_falls_back_to_manual(self, tmp_path: Path):
        raw = [{"start": 0.0, "end": 1.0, "sig": [4, 4], "confidence": 0.5, "source": "some_future_model"}]
        p = tmp_path / "future.json"
        p.write_text(json.dumps(raw))
        loaded = read_time_signatures_json(p)
        assert loaded[0].source == "manual"

    def test_empty_list(self, tmp_path: Path):
        p = tmp_path / "empty.json"
        write_time_signatures_json([], p)
        assert read_time_signatures_json(p) == []

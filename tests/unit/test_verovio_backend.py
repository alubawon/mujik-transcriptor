"""Tests for Verovio backend (real Verovio, simple MusicXML)."""
from __future__ import annotations

import pytest

from mujik.render.verovio_backend import (
    VerovioBackend,
    VerovioBackendError,
    render_musicxml_to_svg,
)


# 一个最简单的 MusicXML 片段
SIMPLE_MUSICXML = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1">
      <part-name>Music</part-name>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>4</duration>
        <type>whole</type>
      </note>
    </measure>
  </part>
</score-partwise>
"""


class TestVerovioBackend:
    def test_init(self):
        backend = VerovioBackend()
        assert backend is not None

    def test_render_a4(self):
        backend = VerovioBackend()
        svg = backend.render(SIMPLE_MUSICXML, page_size="A4", staff_count=1)
        assert isinstance(svg, str)
        assert len(svg) > 100
        # SVG 文件头
        assert "<svg" in svg.lower()

    def test_render_letter(self):
        backend = VerovioBackend()
        svg = backend.render(SIMPLE_MUSICXML, page_size="Letter", staff_count=1)
        assert "<svg" in svg.lower()

    def test_empty_input_raises(self):
        backend = VerovioBackend()
        with pytest.raises(VerovioBackendError, match="empty"):
            backend.render("", page_size="A4")

    def test_whitespace_input_raises(self):
        backend = VerovioBackend()
        with pytest.raises(VerovioBackendError, match="empty"):
            backend.render("   \n  ", page_size="A4")

    def test_invalid_musicxml_raises(self):
        backend = VerovioBackend()
        with pytest.raises(VerovioBackendError, match="parse"):
            backend.render("<not-musicxml/>", page_size="A4")

    def test_custom_options(self):
        backend = VerovioBackend(options={"scale": 75})
        svg = backend.render(SIMPLE_MUSICXML, page_size="A4")
        assert "<svg" in svg.lower()


def test_render_musicxml_to_svg_function():
    svg = render_musicxml_to_svg(SIMPLE_MUSICXML)
    assert "<svg" in svg.lower()


def test_import_error():
    """verovio not installed path."""
    import sys
    original = sys.modules.get("verovio")
    sys.modules["verovio"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(VerovioBackendError, match="verovio is not installed"):
            VerovioBackend()
    finally:
        if original is not None:
            sys.modules["verovio"] = original
        else:
            sys.modules.pop("verovio", None)

"""Tests for mujik render CLI subcommand (v0.2.4: SVG + PDF)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mujik.cli import main


SIMPLE_MUSICXML = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1"><part-name>Music</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <time><beats>4</beats><beat-type>4</beat-type></time>
      </attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><type>whole</type></note>
    </measure>
  </part>
</score-partwise>
"""


def _setup_musicxml(tmp_path: Path) -> Path:
    p = tmp_path / "in.musicxml"
    p.write_text(SIMPLE_MUSICXML, encoding="utf-8")
    return p


class TestRenderCLI:
    def test_default_svg(self, tmp_path: Path):
        """默认 verovio backend → SVG。"""
        in_path = _setup_musicxml(tmp_path)
        out_path = tmp_path / "out"

        # 不真跑 verovio，mock 掉 render_musicxml_to_svg
        # 注意：patch 的目标必须是"调用 lookup 发生的命名空间"，
        # 即 mujik.render，而非 mujik.render.verovio_backend
        with patch("mujik.render.render_musicxml_to_svg",
                   return_value="<svg></svg>"):
            rc = main([
                "render",
                "--input", str(in_path),
                "--output", str(out_path),
            ])

        assert rc == 0
        # 默认输出 .svg
        svg_path = tmp_path / "out.svg"
        assert svg_path.exists()
        assert svg_path.read_text() == "<svg></svg>"

    def test_explicit_svg_extension(self, tmp_path: Path):
        in_path = _setup_musicxml(tmp_path)
        out_path = tmp_path / "explicit.svg"

        with patch("mujik.render.render_musicxml_to_svg",
                   return_value="<svg>x</svg>"):
            rc = main([
                "render", "--input", str(in_path), "--output", str(out_path),
            ])

        assert rc == 0
        assert out_path.exists()

    def test_pdf_flag_uses_cli(self, tmp_path: Path):
        """--pdf 标志：verovio backend 走 CLI subprocess。"""
        in_path = _setup_musicxml(tmp_path)
        out_path = tmp_path / "out"

        def fake_render_pdf(self, musicxml_str, out_path, page_size="A4"):
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"%PDF-1.4\n%fake\n")
            return p

        with patch("mujik.render.verovio_cli.VerovioCliBackend.is_available",
                   return_value=True), \
             patch("mujik.render.verovio_cli.VerovioCliBackend.render_to_pdf",
                   new=fake_render_pdf):
            rc = main([
                "render",
                "--input", str(in_path),
                "--output", str(out_path),
                "--pdf",
            ])

        assert rc == 0
        pdf_path = tmp_path / "out.pdf"
        assert pdf_path.exists()
        assert pdf_path.read_bytes().startswith(b"%PDF-")

    def test_pdf_cli_not_available_falls_back_to_svg_pdf(self, tmp_path: Path):
        """v0.5.2: CLI 不可用不再报错，回退 verovio toolkit SVG→cairosvg→pypdf。"""
        in_path = _setup_musicxml(tmp_path)
        out_path = tmp_path / "out.pdf"

        with patch("mujik.render.verovio_cli.VerovioCliBackend.is_available",
                   return_value=False), \
             patch("mujik.render.verovio_svg_pdf.render_musicxml_to_pdf_via_svg",
                   return_value=out_path) as mock_fallback:
            main([
                "render",
                "--input", str(in_path),
                "--output", str(out_path),
                "--pdf",
            ])
        mock_fallback.assert_called_once()

    def test_lilypond_backend_dispatches(self, tmp_path: Path):
        """--backend lilypond 应走 LilyPondClient（mock）。"""
        in_path = _setup_musicxml(tmp_path)
        out_path = tmp_path / "out.pdf"

        with patch("mujik.render.render_via_lilypond",
                   return_value=b"%PDF-1.4\n%lilypond\n"):
            rc = main([
                "render",
                "--input", str(in_path),
                "--output", str(out_path),
                "--backend", "lilypond",
            ])

        assert rc == 0
        # pdf 路径写盘
        assert out_path.exists()
        assert out_path.read_bytes().startswith(b"%PDF-")

    def test_render_help(self):
        with pytest.raises(SystemExit):
            main(["render", "--help"])

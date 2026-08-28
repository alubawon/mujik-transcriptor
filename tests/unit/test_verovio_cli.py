"""Tests for render/verovio_cli.py (mocked subprocess)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mujik.render.verovio_cli import (
    VerovioCliBackend,
    VerovioCliBackendError,
    render_musicxml_to_pdf,
)


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


def _mock_completed_proc(returncode: int = 0, stderr: str = ""):
    p = MagicMock()
    p.returncode = returncode
    p.stderr = stderr
    p.stdout = ""
    return p


class TestVerovioCliBackend:
    def test_is_available_true(self):
        with patch("mujik.render.verovio_cli.shutil.which", return_value="/usr/bin/verovio"):
            b = VerovioCliBackend()
            assert b.is_available() is True

    def test_is_available_false(self):
        with patch("mujik.render.verovio_cli.shutil.which", return_value=None):
            b = VerovioCliBackend()
            assert b.is_available() is False

    def test_render_to_pdf_success(self, tmp_path: Path):
        out = tmp_path / "out.pdf"

        def fake_run(*args, **kwargs):
            # 模拟 verovio 写出 PDF
            out.write_bytes(b"%PDF-1.4\n%fake\n")
            return _mock_completed_proc(returncode=0)

        with patch("mujik.render.verovio_cli.shutil.which", return_value="/usr/bin/verovio"), \
             patch("mujik.render.verovio_cli.subprocess.run", side_effect=fake_run):
            b = VerovioCliBackend()
            result = b.render_to_pdf(SIMPLE_MUSICXML, out)

        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0
        assert out.read_bytes().startswith(b"%PDF-")

    def test_render_to_svg_success(self, tmp_path: Path):
        out = tmp_path / "out.svg"

        def fake_run(*args, **kwargs):
            out.write_text("<svg></svg>", encoding="utf-8")
            return _mock_completed_proc(returncode=0)

        with patch("mujik.render.verovio_cli.shutil.which", return_value="/usr/bin/verovio"), \
             patch("mujik.render.verovio_cli.subprocess.run", side_effect=fake_run):
            b = VerovioCliBackend()
            result = b.render_to_svg(SIMPLE_MUSICXML, out)

        assert result == out
        assert out.exists()

    def test_cli_not_available_raises(self, tmp_path: Path):
        with patch("mujik.render.verovio_cli.shutil.which", return_value=None):
            b = VerovioCliBackend()
            with pytest.raises(VerovioCliBackendError, match="not found"):
                b.render_to_pdf(SIMPLE_MUSICXML, tmp_path / "out.pdf")

    def test_cli_nonzero_returncode(self, tmp_path: Path):
        with patch("mujik.render.verovio_cli.shutil.which", return_value="/usr/bin/verovio"), \
             patch("mujik.render.verovio_cli.subprocess.run",
                   return_value=_mock_completed_proc(returncode=1, stderr="parse error")):
            b = VerovioCliBackend()
            with pytest.raises(VerovioCliBackendError, match="rc=1"):
                b.render_to_pdf(SIMPLE_MUSICXML, tmp_path / "out.pdf")

    def test_cli_timeout(self, tmp_path: Path):
        with patch("mujik.render.verovio_cli.shutil.which", return_value="/usr/bin/verovio"), \
             patch("mujik.render.verovio_cli.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="verovio", timeout=10)):
            b = VerovioCliBackend(timeout_sec=10)
            with pytest.raises(VerovioCliBackendError, match="timed out"):
                b.render_to_pdf(SIMPLE_MUSICXML, tmp_path / "out.pdf")

    def test_empty_input_raises(self, tmp_path: Path):
        with patch("mujik.render.verovio_cli.shutil.which", return_value="/usr/bin/verovio"):
            b = VerovioCliBackend()
            with pytest.raises(VerovioCliBackendError, match="empty"):
                b.render_to_pdf("", tmp_path / "out.pdf")

    def test_no_output_produced(self, tmp_path: Path):
        with patch("mujik.render.verovio_cli.shutil.which", return_value="/usr/bin/verovio"), \
             patch("mujik.render.verovio_cli.subprocess.run",
                   return_value=_mock_completed_proc(returncode=0)):
            b = VerovioCliBackend()
            with pytest.raises(VerovioCliBackendError, match="did not produce"):
                b.render_to_pdf(SIMPLE_MUSICXML, tmp_path / "out.pdf")


class TestConvenienceFunction:
    def test_render_musicxml_to_pdf(self, tmp_path: Path):
        out = tmp_path / "doc.pdf"

        def fake_run(*args, **kwargs):
            out.write_bytes(b"%PDF-1.4")
            return _mock_completed_proc(returncode=0)

        with patch("mujik.render.verovio_cli.shutil.which", return_value="/usr/bin/verovio"), \
             patch("mujik.render.verovio_cli.subprocess.run", side_effect=fake_run):
            result = render_musicxml_to_pdf(SIMPLE_MUSICXML, out)
        assert result == out

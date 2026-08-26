"""Tests for render HTTP clients (lilypond_client, musescore_client, render/__init__.py)."""
from __future__ import annotations

import base64
from unittest.mock import patch, MagicMock

import httpx
import pytest

from mujik.render.lilypond_client import (
    LilyPondClient,
    LilyPondClientError,
    render_via_lilypond,
)
from mujik.render.musescore_client import (
    MuseScoreClient,
    MuseScoreClientError,
    render_via_musescore,
)
from mujik.render import render_musicxml, render_musicxml_to_file


SIMPLE_MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>X</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
      </attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
    </measure>
  </part>
</score-partwise>
"""


# ----- LilyPondClient -----

class TestLilyPondClient:
    def test_health(self):
        with patch("httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=MagicMock(return_value={"status": "ok", "lilypond_version": "2.24"}),
                raise_for_status=MagicMock(),
            )
            result = LilyPondClient().health()
            assert result["status"] == "ok"

    def test_health_failure(self):
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(LilyPondClientError, match="health check"):
                LilyPondClient().health()

    def test_render_success(self):
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=MagicMock(return_value={"pdf_b64": base64.b64encode(b"%PDF-1.4").decode()}),
            )
            pdf = LilyPondClient().render(SIMPLE_MUSICXML)
            assert pdf == b"%PDF-1.4"

    def test_render_with_midi_bytes(self):
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=MagicMock(return_value={"pdf_b64": base64.b64encode(b"%PDF").decode()}),
            )
            pdf = LilyPondClient().render(b"\x00\x01\x02", input_type="midi")
            assert pdf == b"%PDF"
            # 确认 input_type 正确传递
            call_json = mock_post.call_args.kwargs["json"]
            assert call_json["input_type"] == "midi"

    def test_render_http_error(self):
        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(LilyPondClientError, match="request failed"):
                LilyPondClient().render(SIMPLE_MUSICXML)

    def test_render_status_error(self):
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=500,
                text="lilypond crashed",
            )
            with pytest.raises(LilyPondClientError, match="render failed"):
                LilyPondClient().render(SIMPLE_MUSICXML)

    def test_render_invalid_response(self):
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=MagicMock(return_value={"other": "field"}),
            )
            with pytest.raises(LilyPondClientError, match="invalid response"):
                LilyPondClient().render(SIMPLE_MUSICXML)


def test_render_via_lilypond_function():
    with patch("httpx.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"pdf_b64": base64.b64encode(b"X").decode()}),
        )
        cfg = RenderConfig(lilypond_url="http://lilypond:5001")
        pdf = render_via_lilypond(SIMPLE_MUSICXML, config=cfg)
        assert pdf == b"X"
        # URL is taken from config
        call_url = mock_post.call_args[0][0]
        assert "lilypond:5001" in call_url


# ----- MuseScoreClient -----

class TestMuseScoreClient:
    def test_health(self):
        with patch("httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=MagicMock(return_value={"status": "ok", "musescore_version": "4.0"}),
                raise_for_status=MagicMock(),
            )
            result = MuseScoreClient().health()
            assert result["status"] == "ok"

    def test_render_success(self):
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(
                status_code=200,
                json=MagicMock(return_value={"pdf_b64": base64.b64encode(b"%PDF").decode()}),
            )
            pdf = MuseScoreClient().render(SIMPLE_MUSICXML)
            assert pdf == b"%PDF"


def test_render_via_musescore_function():
    with patch("httpx.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"pdf_b64": base64.b64encode(b"X").decode()}),
        )
        cfg = RenderConfig(musescore_url="http://musescore:5002")
        pdf = render_via_musescore(SIMPLE_MUSICXML, config=cfg)
        assert pdf == b"X"


# ----- render_musicxml unified entry -----

class TestRenderMusicxml:
    def test_verovio_backend(self):
        cfg = RenderConfig(pdf_backend="verovio")
        result = render_musicxml(SIMPLE_MUSICXML, config=cfg)
        assert isinstance(result, bytes)
        assert b"<svg" in result.lower()

    def test_unknown_backend_raises(self):
        # 绕过 pydantic 验证，构造非法值
        cfg = RenderConfig()
        object.__setattr__(cfg, "pdf_backend", "nonexistent")
        with pytest.raises(ValueError, match="unknown pdf_backend"):
            render_musicxml(SIMPLE_MUSICXML, config=cfg)

    def test_lilypond_backend_dispatches(self):
        with patch("mujik.render.render_via_lilypond") as mock_lp:
            mock_lp.return_value = b"%PDF"
            cfg = RenderConfig(pdf_backend="lilypond")
            result = render_musicxml(SIMPLE_MUSICXML, config=cfg)
            assert result == b"%PDF"
            mock_lp.assert_called_once()

    def test_musescore_backend_dispatches(self):
        with patch("mujik.render.render_via_musescore") as mock_ms:
            mock_ms.return_value = b"%PDF"
            cfg = RenderConfig(pdf_backend="musescore")
            result = render_musicxml(SIMPLE_MUSICXML, config=cfg)
            assert result == b"%PDF"


class TestRenderMusicxmlToFile:
    def test_verovio_writes_svg(self, tmp_path):
        out = tmp_path / "score"
        cfg = RenderConfig(pdf_backend="verovio")
        result = render_musicxml_to_file(SIMPLE_MUSICXML, str(out), config=cfg)
        assert result.endswith(".svg")
        content = Path(result).read_text()
        assert "<svg" in content.lower()

    def test_lilypond_writes_pdf(self, tmp_path):
        out = tmp_path / "score"
        with patch("mujik.render.render_via_lilypond") as mock_lp:
            mock_lp.return_value = b"%PDF-1.4"
            cfg = RenderConfig(pdf_backend="lilypond")
            result = render_musicxml_to_file(SIMPLE_MUSICXML, str(out), config=cfg)
            assert result.endswith(".pdf")
            content = Path(result).read_bytes()
            assert content == b"%PDF-1.4"

    def test_explicit_svg_extension_kept(self, tmp_path):
        out = tmp_path / "score.svg"
        cfg = RenderConfig(pdf_backend="verovio")
        result = render_musicxml_to_file(SIMPLE_MUSICXML, str(out), config=cfg)
        assert result == str(out)


from pathlib import Path

from mujik.config.schema import RenderConfig
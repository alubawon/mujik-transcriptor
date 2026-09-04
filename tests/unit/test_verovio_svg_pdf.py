"""Tests for render/verovio_svg_pdf.py (v0.5.2 SVG→PDF 主路径)。"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mujik.render.verovio_backend import VerovioBackend, VerovioBackendError
from mujik.render.verovio_svg_pdf import (
    VerovioSvgPdfError,
    check_svg_pdf_available,
    render_musicxml_to_pdf_via_svg,
)

from tests.unit.test_verovio_backend import SIMPLE_MUSICXML


def _require_libcairo() -> None:
    """cairosvg 装了但系统 libcairo 缺失（macOS 宿主机 / CI）时跳过。"""
    try:
        import cairosvg  # noqa: F401
    except OSError as e:
        pytest.skip(f"system libcairo2 unavailable: {e}")


class TestCheckAvailable:
    def test_all_present(self):
        with patch.dict("sys.modules", {
            "verovio": MagicMock(), "cairosvg": MagicMock(), "pypdf": MagicMock(),
        }):
            assert check_svg_pdf_available() is True

    def test_cairosvg_missing(self):
        with patch.dict("sys.modules", {
            "verovio": MagicMock(), "cairosvg": None, "pypdf": MagicMock(),
        }):
            assert check_svg_pdf_available() is False

    def test_pypdf_missing(self):
        with patch.dict("sys.modules", {
            "verovio": MagicMock(), "cairosvg": MagicMock(), "pypdf": None,
        }):
            assert check_svg_pdf_available() is False


class TestRenderPages:
    def test_render_pages_multi_page(self):
        """VerovioBackend.render_pages 返回 getPageCount() 页 SVG。"""
        backend = VerovioBackend()
        svgs = backend.render_pages(SIMPLE_MUSICXML)
        assert len(svgs) >= 1
        assert all("<svg" in s for s in svgs)

    def test_render_pages_empty_input(self):
        backend = VerovioBackend()
        with pytest.raises(VerovioBackendError, match="empty MusicXML"):
            backend.render_pages("   ")


class TestRenderPdfViaSvg:
    def test_end_to_end_small_score(self, tmp_path: Path):
        """真实小谱：verovio → cairosvg → 合法 PDF。缺系统 libcairo2 时跳过。"""
        pytest.importorskip("pypdf")
        pytest.importorskip("verovio")
        _require_libcairo()
        out = tmp_path / "score.pdf"
        result = render_musicxml_to_pdf_via_svg(SIMPLE_MUSICXML, out)
        assert result == out
        assert out.read_bytes().startswith(b"%PDF")

    def test_missing_dep_raises(self, tmp_path: Path):
        out = tmp_path / "score.pdf"
        with patch.dict("sys.modules", {"cairosvg": None}):
            with pytest.raises(VerovioSvgPdfError, match="missing dep"):
                render_musicxml_to_pdf_via_svg(SIMPLE_MUSICXML, out)

    def test_verovio_parse_failure_wrapped(self, tmp_path: Path):
        pytest.importorskip("pypdf")
        pytest.importorskip("verovio")
        _require_libcairo()
        out = tmp_path / "score.pdf"
        with pytest.raises(VerovioSvgPdfError, match="verovio SVG render failed"):
            render_musicxml_to_pdf_via_svg("not musicxml at all", out)

    def test_appends_pdf_suffix_via_caller(self, tmp_path: Path):
        """本模块不擅自改后缀（suffix 逻辑在 render/__init__）；路径原样使用。"""
        pytest.importorskip("pypdf")
        pytest.importorskip("verovio")
        _require_libcairo()
        out = tmp_path / "score"  # 无 .pdf 后缀
        result = render_musicxml_to_pdf_via_svg(SIMPLE_MUSICXML, out)
        assert result == out
        assert out.read_bytes().startswith(b"%PDF")

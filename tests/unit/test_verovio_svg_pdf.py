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


class TestInlineMusicFontGlyphs:
    """v0.5.3：verovio 的 <tspan font-family="Leipzig">SMuFL</tspan> → 内嵌 path。

    cairosvg 走系统字体查不到 Leipzig（Docker/macOS 默认没装）→ metronome
    beat-unit 字符（♩）渲染成豆腐块。后处理用 verovio 自带 data/Leipzig/
    的字形 path 就地替换，零系统字体依赖。
    """

    # verovio tempo direction 的实际输出形态（buhee page1 实测）
    TEMPO_SVG = (
        '<g id="t1" class="tempo">'
        '<text x="2188" y="1209" font-size="0px">'
        '<tspan class="rend"><tspan class="text">'
        '<tspan font-family="Leipzig" font-size="720px"></tspan>'
        "</tspan></tspan>"
        '<tspan class="text"><tspan font-size="405px">\xa0=\xa0</tspan></tspan>'
        '<tspan class="text"><tspan font-size="405px">125.0</tspan></tspan>'
        "</text></g>"
    )

    def test_leipzig_tspan_replaced_with_path(self):
        from mujik.render.verovio_svg_pdf import _inline_music_font_glyphs

        out = _inline_music_font_glyphs(self.TEMPO_SVG)
        assert 'font-family="Leipzig"' not in out
        # 字形 path + font-unit→px 缩放（720px / 1000 upem = 0.72）
        assert '<path fill="#000000"' in out
        assert "translate(2188.00,1209.00) scale(0.720000,-0.720000)" in out
        # 数字部分保留为独立 <text>，x = 2188 + advance*0.72（ECA5 advance=302）
        assert '<text x="2405.44" y="1209.00" font-size="405px">' in out
        assert "\xa0=\xa0125.0" in out

    def test_glyph_resolved_from_verovio_package(self):
        from mujik.render.verovio_svg_pdf import _leipzig_glyph

        # ECA5 = metNoteQuarterUp（buhee 实测字符）
        glyph = _leipzig_glyph(0xECA5)
        assert glyph is not None
        path_d, advance = glyph
        assert path_d.startswith("M")
        assert advance > 0

    def test_non_pua_leipzig_text_untouched(self):
        from mujik.render.verovio_svg_pdf import _inline_music_font_glyphs

        svg = '<text x="10" y="20"><tspan font-family="Leipzig" font-size="100px">abc</tspan></text>'
        assert _inline_music_font_glyphs(svg) == svg

    def test_missing_glyph_keeps_block_and_warns(self):

        from mujik.render.verovio_svg_pdf import _inline_music_font_glyphs

        svg = (
            '<text x="1" y="2" font-size="0px">'
            '<tspan font-family="Leipzig" font-size="100px">蓮</tspan>'
            "</text>"
        )
        assert _inline_music_font_glyphs(svg) == svg

    def test_plain_text_block_untouched(self):
        from mujik.render.verovio_svg_pdf import _inline_music_font_glyphs

        svg = '<text x="5" y="6" font-size="405px">125.0</text>'
        assert _inline_music_font_glyphs(svg) == svg

"""MusicXML → PDF（verovio toolkit 出 SVG → cairosvg 转 PDF，v0.5.2）。

Verovio CLI 在 Debian/Ubuntu apt 源里没有包（只有 brew / npm），Docker 镜像
装不上；而 Verovio Python binding 只有 renderToSVG()。本模块是镜像内的
PDF 主路径：

    MusicXML --verovio toolkit--> 每页 SVG --cairosvg--> 每页 PDF --pypdf--> 合并

音符符头等音乐字形（SMuFL）在 Verovio SVG 里是 path 而非 <text>，无需安装
音乐字体；但速度记号的 beat-unit 字符（♩ 等，metNote*）verovio 输出为
``<tspan font-family="Leipzig">私有区字符</tspan>``——cairosvg 走系统字体
查不到 Leipzig（Docker/macOS 默认都没装）→ 豆腐块。v0.5.3 起
:func:`_inline_music_font_glyphs` 在 svg2pdf 前把这些 tspan 就地替换成
verovio 自带 ``data/Leipzig/`` 的字形 path 轮廓，零系统字体依赖。

依赖：verovio + cairosvg + pypdf（均在 render extra）。
"""
from __future__ import annotations

import re
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

from loguru import logger


class VerovioSvgPdfError(RuntimeError):
    pass


# SMuFL 私有区（PUA）起点：verovio 把 <metronome> beat-unit 等以
# <tspan font-family="Leipzig" font-size="Npx">U+Exxx</tspan> 输出
_SMUFL_PUA_START = 0xE000
_LEIPZIG_TSPAN_RE = re.compile(
    r'<tspan font-family="Leipzig" font-size="([\d.]+)px">([^<]+)</tspan>'
)
# 含 Leipzig tspan 的整个 <text> 块（verovio tempo/dynam 的固定结构）
_LEIPZIG_TEXT_RE = re.compile(r"<text[^>]*>.*?</text>", re.DOTALL)
_TEXT_XY_RE = re.compile(r'<text x="([-\d.]+)" y="([-\d.]+)"')
_TSPAN_SIZE_RE = re.compile(r'font-size="([\d.]+)px"')


def _leipzig_data_dir() -> Path:
    """verovio 包自带的 Leipzig 字形目录（per-glyph SVG path xml）。"""
    import verovio

    return Path(verovio.__file__).parent / "data" / "Leipzig"


@lru_cache(maxsize=1)
def _leipzig_metrics() -> dict[int, float]:
    """codepoint → 水平 advance（font units），来自 Leipzig.xml bounding-boxes。"""
    metrics: dict[int, float] = {}
    bounds = _leipzig_data_dir().parent / "Leipzig.xml"
    if bounds.is_file():
        for m in re.finditer(r'<g c="([0-9A-F]{4})"[^>]*?h-a-x="([\d.]+)"', bounds.read_text()):
            metrics[int(m.group(1), 16)] = float(m.group(2))
    return metrics


@lru_cache(maxsize=1024)
def _leipzig_glyph(codepoint: int) -> tuple[str, float] | None:
    """codepoint → (path d（font units，y-up，原文件内的 scale(1,-1) 不含）,
    advance font units)。找不到字形返回 None。"""
    glyph_file = _leipzig_data_dir() / f"{codepoint:04X}.xml"
    if not glyph_file.is_file():
        return None
    m = re.search(r'\bd="([^"]+)"', glyph_file.read_text())
    if not m:
        return None
    return m.group(1), _leipzig_metrics().get(codepoint, 500.0)


def _inline_music_font_glyphs(svg: str) -> str:
    """把 verovio SVG 里 font-family="Leipzig" 的 SMuFL 文本 tspan 内联为 path。

    只处理含恰好一个 Leipzig tspan 的 <text> 块（metronome/tempo 的实际
    形态：♩ 在前、'=125.0' 数字在后）。多个 Leipzig tspan 的块保持原样并
    打 warning（未出现过的形态，fail-soft 保留 tofu 优于渲染错位）。
    """
    def _process_text_block(m: re.Match) -> str:
        block = m.group(0)
        tspans = _LEIPZIG_TSPAN_RE.findall(block)
        if not tspans:
            return block
        if len(tspans) > 1:
            logger.warning(
                "svg-font-inline: {n} Leipzig tspans in one <text>, skipping "
                "(unhandled verovio shape)",
                n=len(tspans),
            )
            return block

        xy = _TEXT_XY_RE.search(block)
        if not xy:
            return block
        x, y = float(xy.group(1)), float(xy.group(2))

        size_str, content = tspans[0]
        font_size = float(size_str)
        scale = font_size / 1000.0  # Leipzig units-per-em=1000

        # 去掉所有 tspan 标签后的纯文本内容，去掉首字符即数字部分
        plain = re.sub(r"</?tspan[^>]*>", "", block)
        plain = re.sub(r"</?text[^>]*>", "", plain).lstrip()
        if not plain:
            return block
        first_cp = ord(plain[0])
        glyph = _leipzig_glyph(first_cp)
        if glyph is None:
            logger.warning(
                "svg-font-inline: Leipzig glyph U+{cp:04X} not found in "
                "verovio data (font not installed either) — glyph dropped",
                cp=first_cp,
            )
            return block
        path_d, advance = glyph

        # 数字部分沿用其余 tspan 的字号（'=125.0' 等）
        rest = plain[1:]
        rest_sizes = [
            float(s) for s in _TSPAN_SIZE_RE.findall(block)
            if abs(float(s) - font_size) > 1e-6
        ]
        rest_size = max(rest_sizes) if rest_sizes else font_size

        glyph_x = x
        rest_x = x + advance * scale
        # 字形 path 是 font-units y-up；translate+scale(s,-s) 映射到 SVG
        # y-down（原文件自带的 scale(1,-1) 是给 y-down 渲染器翻 y 用的，
        # 这里不用它，直接对原始 d 做翻转）
        replacement = (
            f'<path fill="#000000" transform="translate({glyph_x:.2f},{y:.2f}) '
            f'scale({scale:.6f},-{scale:.6f})" d="{path_d}"/>'
            f'<text x="{rest_x:.2f}" y="{y:.2f}" font-size="{rest_size:g}px">'
            f"{rest}</text>"
        )
        return replacement

    return _LEIPZIG_TEXT_RE.sub(_process_text_block, svg)


def check_svg_pdf_available() -> bool:
    """检查 SVG→PDF 依赖是否可用（verovio + cairosvg + pypdf）。"""
    try:
        import cairosvg  # noqa: F401
        import pypdf  # noqa: F401
        import verovio  # noqa: F401
        return True
    except ImportError:
        return False


def render_musicxml_to_pdf_via_svg(
    musicxml_str: str,
    out_path: str | Path,
    page_size: Literal["A4", "Letter"] = "A4",
) -> Path:
    """MusicXML → PDF，纯 Python 路径（verovio toolkit → cairosvg → pypdf）。

    Args:
        musicxml_str: MusicXML 内容
        out_path: 输出 .pdf 路径
        page_size: 页面尺寸

    Returns:
        实际写入的路径

    Raises:
        VerovioSvgPdfError: 依赖缺失 / verovio 解析失败 / 页面渲染失败
    """
    out_path = Path(out_path)

    try:
        import cairosvg
        from pypdf import PdfWriter
    except ImportError as e:
        raise VerovioSvgPdfError(
            f"missing dep: {e}; install via `uv pip install mujik-transcriptor[render]` "
            f"(and system libcairo2)"
        ) from e

    from mujik.render.verovio_backend import VerovioBackend, VerovioBackendError

    try:
        backend = VerovioBackend()
        svgs = backend.render_pages(musicxml_str, page_size=page_size)
    except VerovioBackendError as e:
        raise VerovioSvgPdfError(f"verovio SVG render failed: {e}") from e

    # 原子性：每页先写 tmp，全部成功后合并 rename 到目标
    with tempfile.TemporaryDirectory(prefix="mujik_svg_pdf_") as tmp:
        page_pdfs: list[Path] = []
        for i, svg in enumerate(svgs, start=1):
            page_pdf = Path(tmp) / f"page_{i}.pdf"
            try:
                # v0.5.3: metronome 等 Leipzig 字体文本 → 内嵌 path（免系统字体）
                svg = _inline_music_font_glyphs(svg)
                cairosvg.svg2pdf(bytestring=svg.encode("utf-8"),
                                 write_to=str(page_pdf))
            except Exception as e:
                raise VerovioSvgPdfError(
                    f"cairosvg svg->pdf failed on page {i}/{len(svgs)}: {e}"
                ) from e
            page_pdfs.append(page_pdf)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_out = Path(tmp) / "merged.pdf"
        try:
            writer = PdfWriter()
            for page_pdf in page_pdfs:
                writer.append(str(page_pdf))
            with open(tmp_out, "wb") as f:
                writer.write(f)
        except Exception as e:
            raise VerovioSvgPdfError(f"pypdf merge failed: {e}") from e
        # shutil.move 而非 Path.replace：容器里 tmp（tmpfs）与输出（bind mount）
        # 常跨文件系统，os.replace 会报 Invalid cross-device link
        shutil.move(str(tmp_out), str(out_path))

    logger.info(
        "Rendered: backend=verovio-svg-pdf, pages={n}, output={out}",
        n=len(svgs), out=out_path,
    )
    return out_path


__all__ = [
    "VerovioSvgPdfError",
    "check_svg_pdf_available",
    "render_musicxml_to_pdf_via_svg",
]

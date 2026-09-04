"""MusicXML → PDF（verovio toolkit 出 SVG → cairosvg 转 PDF，v0.5.2）。

Verovio CLI 在 Debian/Ubuntu apt 源里没有包（只有 brew / npm），Docker 镜像
装不上；而 Verovio Python binding 只有 renderToSVG()。本模块是镜像内的
PDF 主路径：

    MusicXML --verovio toolkit--> 每页 SVG --cairosvg--> 每页 PDF --pypdf--> 合并

音乐字形（SMuFL）在 Verovio SVG 里是 path 而非 <text>，因此无需安装
音乐字体；系统只需 libcairo2（Dockerfile.ml: apt install libcairo2）。

依赖：verovio + cairosvg + pypdf（均在 render extra）。
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Literal

from loguru import logger


class VerovioSvgPdfError(RuntimeError):
    pass


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

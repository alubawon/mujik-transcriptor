"""Verovio 渲染后端（BSD，主线内嵌，v0.1 输出 SVG）。

Verovio 6.x 的 Python 工具包只提供 `renderToSVG()`，不提供 `renderToPDF()`。
本模块负责 MusicXML → SVG；PDF 输出走 GPL 隔离的 LilyPond / MuseScore 服务。

调用方式：
    backend = VerovioBackend()
    svg_str = backend.render(musicxml_str, page_size="A4", staff_count=2)
"""
from __future__ import annotations

from typing import Literal

from loguru import logger

from mujik.config.schema import RenderConfig


class VerovioBackendError(RuntimeError):
    pass


class VerovioBackend:
    """Verovio 渲染后端包装（v0.1：SVG 输出）。"""

    def __init__(self, options: dict | None = None) -> None:
        """初始化 Verovio toolkit。

        Args:
            options: Verovio 选项字典，覆盖默认。
        """
        try:
            import verovio
        except ImportError as e:
            raise VerovioBackendError(
                "verovio is not installed; install via `uv pip install mujik-transcriptor[render]`"
            ) from e

        self._verovio = verovio
        self._tk = verovio.toolkit()

        default_options = {
            "pageHeight": 2970,   # A4 (mm * 10)
            "pageWidth": 2100,
            "pageMarginTop": 50,
            "pageMarginBottom": 50,
            "pageMarginLeft": 50,
            "pageMarginRight": 50,
            "scale": 100,
            "adjustPageHeight": True,
            "breaks": "auto",
            "spacingStaff": 4,
            "svgBoundingBoxes": False,
            "svgViewBox": True,
        }
        if options:
            default_options.update(options)
        self._tk.setOptions(default_options)
        logger.debug("Verovio initialized with options: {opts}", opts=default_options)

    def render(
        self,
        musicxml_str: str,
        page_size: Literal["A4", "Letter"] = "A4",
        staff_count: int = 2,
    ) -> str:
        """从 MusicXML 字符串渲染 SVG。

        Args:
            musicxml_str: MusicXML 内容
            page_size: 页面尺寸
            staff_count: 谱表行数（参考值，用于日志）

        Returns:
            SVG 字符串

        Raises:
            VerovioBackendError: 渲染失败
        """
        self._set_page_size(page_size)

        if not musicxml_str or not musicxml_str.strip():
            raise VerovioBackendError("empty MusicXML input")

        success = self._tk.loadData(musicxml_str)
        if not success:
            raise VerovioBackendError("Verovio failed to parse MusicXML")

        try:
            svg_str: str = self._tk.renderToSVG()
        except Exception as e:
            raise VerovioBackendError(f"Verovio renderToSVG failed: {e}") from e

        if not svg_str:
            raise VerovioBackendError("Verovio returned empty SVG")

        logger.info(
            "Verovio rendered SVG: {n} chars, staff_count={k}",
            n=len(svg_str), k=staff_count,
        )
        return svg_str

    def _set_page_size(self, page_size: str) -> None:
        if page_size == "A4":
            self._tk.setOptions({"pageHeight": 2970, "pageWidth": 2100})
        elif page_size == "Letter":
            self._tk.setOptions({"pageHeight": 2794, "pageWidth": 2159})


def render_musicxml_to_svg(
    musicxml_str: str,
    config: RenderConfig | None = None,
) -> str:
    """便捷函数：MusicXML → SVG。"""
    cfg = config or RenderConfig()
    backend = VerovioBackend()
    return backend.render(
        musicxml_str,
        page_size=cfg.page_size,
        staff_count=cfg.staff_count,
    )


__all__ = [
    "VerovioBackend",
    "VerovioBackendError",
    "render_musicxml_to_svg",
]

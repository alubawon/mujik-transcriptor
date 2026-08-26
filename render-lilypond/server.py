"""LilyPond 渲染服务（GPL-2.0+）。

接收 MusicXML/MIDI，调用 LilyPond CLI 编译为 PDF，返回 base64。
"""
from __future__ import annotations

import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="mujik-render-lilypond", version="0.1.0")


class RenderRequest(BaseModel):
    input_type: Literal["musicxml", "midi"]
    input_b64: str
    options: dict = {}


class RenderResponse(BaseModel):
    pdf_b64: str
    musicxml_out: str | None = None


class HealthResponse(BaseModel):
    status: str
    lilypond_version: str | None = None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """健康检查。"""
    try:
        result = subprocess.run(
            ["lilypond", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = result.stdout.split("\n")[0] if result.stdout else None
        return HealthResponse(status="ok", lilypond_version=version)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return HealthResponse(status="error: " + str(e), lilypond_version=None)


@app.post("/render", response_model=RenderResponse)
def render(req: RenderRequest) -> RenderResponse:
    """MusicXML/MIDI → PDF。"""
    try:
        input_bytes = base64.b64decode(req.input_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64: {e}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        if req.input_type == "musicxml":
            input_file = tmp / "input.musicxml"
            input_file.write_bytes(input_bytes)
        else:  # midi
            input_file = tmp / "input.midi"
            input_file.write_bytes(input_bytes)

        output_pdf = tmp / "output.pdf"

        # LilyPond CLI: lilypond -o output input.musicxml
        cmd = [
            "lilypond",
            "--pdf",
            "-o", str(tmp / "output"),
            str(input_file),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired as e:
            raise HTTPException(status_code=504, detail="lilypond timeout") from e

        if result.returncode != 0 or not output_pdf.exists():
            raise HTTPException(
                status_code=500,
                detail=f"lilypond failed: {result.stderr[:500]}",
            )

        pdf_bytes = output_pdf.read_bytes()
        return RenderResponse(pdf_b64=base64.b64encode(pdf_bytes).decode())

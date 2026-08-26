"""MuseScore 渲染服务（GPL-2.0+）。

接收 MusicXML/MIDI，调用 MuseScore CLI 导出 PDF，返回 base64。
"""
from __future__ import annotations

import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="mujik-render-musescore", version="0.1.0")


class RenderRequest(BaseModel):
    input_type: Literal["musicxml", "midi"]
    input_b64: str
    options: dict = {}


class RenderResponse(BaseModel):
    pdf_b64: str
    musicxml_out: str | None = None


class HealthResponse(BaseModel):
    status: str
    musescore_version: str | None = None


def _find_musescore() -> str:
    """在 PATH 中查找 MuseScore 可执行文件。"""
    for name in ("mscore", "mscore3", "mscore4", "musescore"):
        try:
            subprocess.run([name, "--version"], capture_output=True, timeout=5)
            return name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    raise FileNotFoundError("MuseScore not found in PATH")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """健康检查。"""
    try:
        mscore = _find_musescore()
        result = subprocess.run(
            [mscore, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = result.stdout.split("\n")[0] if result.stdout else None
        return HealthResponse(status="ok", musescore_version=version)
    except Exception as e:
        return HealthResponse(status="error: " + str(e), musescore_version=None)


@app.post("/render", response_model=RenderResponse)
def render(req: RenderRequest) -> RenderResponse:
    """MusicXML/MIDI → PDF。"""
    try:
        input_bytes = base64.b64decode(req.input_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64: {e}")

    try:
        mscore = _find_musescore()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        if req.input_type == "musicxml":
            input_file = tmp / "input.mscx"
            input_file.write_bytes(input_bytes)
        else:
            input_file = tmp / "input.mid"
            input_file.write_bytes(input_bytes)

        output_pdf = tmp / "input.pdf"

        # MuseScore CLI: mscore -o output.pdf input.mscx
        cmd = [
            mscore,
            "-o", str(output_pdf),
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
            raise HTTPException(status_code=504, detail="musescore timeout") from e

        if result.returncode != 0 or not output_pdf.exists():
            raise HTTPException(
                status_code=500,
                detail=f"musescore failed: {result.stderr[:500]}",
            )

        pdf_bytes = output_pdf.read_bytes()
        return RenderResponse(pdf_b64=base64.b64encode(pdf_bytes).decode())

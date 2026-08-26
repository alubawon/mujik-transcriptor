# render-lilypond

mujik-transcriptor 的 **LilyPond 渲染服务**（GPL 隔离）。

## 许可证

**本子项目许可证：GPL-2.0+**

本服务依赖 LilyPond（LilyPond 是 GPL-2.0+），依据 GPL 的"传染性"条款，**整个子项目**必须以 GPL 许可证发布。**mujik-transcriptor 主项目（`mujik-core/`）保持 MIT**，二者通过 HTTP API + MusicXML/MIDI 中间格式通信，不构成单一作品。

## API

```
POST /render
Content-Type: application/json

{
  "input_type": "musicxml",  // or "midi"
  "input_b64": "<base64 of MusicXML/MIDI>",
  "options": {
    "page_size": "A4",
    "staff_count": 2,
    "include_dynamics": true,
    "include_chords": true
  }
}

→ 200 OK
{
  "pdf_b64": "<base64 PDF>",
  "musicxml_out": "<optional MusicXML for chaining>"
}

→ 503 if LilyPond CLI fails
```

## 本地运行

```bash
# 方式 1：Docker
docker-compose --profile lilypond up render-lilypond

# 方式 2：本地
brew install lilypond
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 5001
```

## 主线调用

```python
from mujik.render.lilypond_client import LilyPondClient
client = LilyPondClient(base_url="http://localhost:5001")
pdf = client.render(musicxml_str, options={"page_size": "A4"})
```

详见 [docs/design.md §6.2](../docs/design.md#62-渲染服务-http-接口)。

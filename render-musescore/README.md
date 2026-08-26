# render-musescore

mujik-transcriptor 的 **MuseScore 渲染服务**（GPL 隔离）。

## 许可证

**本子项目许可证：GPL-2.0+**

本服务依赖 MuseScore（MuseScore 是 GPL-2.0+），依据 GPL 的"传染性"条款，**整个子项目**必须以 GPL 许可证发布。**mujik-transcriptor 主项目（`mujik-core/`）保持 MIT**，二者通过 HTTP API + MusicXML/MIDI 中间格式通信。

## API

与 `render-lilypond` 完全一致（端口 5002）：

```
POST http://localhost:5002/render
Content-Type: application/json

{
  "input_type": "musicxml" | "midi",
  "input_b64": "<base64>",
  "options": { ... }
}
→ { "pdf_b64": "<base64 PDF>" }
```

## 本地运行

```bash
# 方式 1：Docker
docker-compose --profile musescore up render-musescore

# 方式 2：本地
brew install musescore
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 5002
```

## 主线调用

```python
from mujik.render.musescore_client import MuseScoreClient
client = MuseScoreClient(base_url="http://localhost:5002")
pdf = client.render(musicxml_str)
```

详见 [docs/design.md §6.2](../docs/design.md#62-渲染服务-http-接口)。

# mujik-transcriptor

> 端到端音乐音频 → MIDI + PDF 乐谱处理管线
> 目标：流行/摇滚/民谣 + 爵士/复杂和声 + 金属/复杂节奏
> 全开源 / GPU 离线 / 保留演奏细节

## 状态

🚧 **v0.1（开发中）**——地基阶段，详见 [路线图](docs/design.md#10-路线图)

## 文档

- [docs/research.md](docs/research.md) — 选型研究报告
- [docs/design.md](docs/design.md) — 设计文档

## 架构一览

```
WAV/FLAC/MP3
  → pyloudnorm (响度归一)
  → Demucs v4 htdemucs_ft (4-stem 源分离)
  → basic-pitch / adtof (各轨 MIDI 转录)
  → Beat Transformer (节拍/下拍)
  → mido + pretty-midi + music21 (量化/对齐/合并)
  → Verovio (BSD) / LilyPond 隔离服务 (GPL) / MuseScore 隔离服务 (GPL)
  → MIDI + PDF
```

详见 [设计文档 §2](docs/design.md#2-架构总览)。

## 当前选型基线

| 环节 | 首选 | 状态 |
|---|---|---|
| 源分离 | Demucs v4 htdemucs_ft | ⭐ 3-0 验证 |
| 旋律/贝斯/other 转录 | basic-pitch | 🧪 谐波幻觉待工程缓解 |
| 鼓转录 | adtof | 🧪 金属未验证 |
| 节拍/下拍 | Beat Transformer | ⭐ 3-0 验证 |
| 时间签名 | ResNet18/METER2800 | ⚠️ 限 4 类 |
| MIDI 工具 | pretty-midi + mido + music21 | ✅ |
| 渲染主线 | Verovio (BSD) | ✅ |
| 渲染精细 | LilyPond (GPL 隔离) | 🧪 v0.3 |
| 渲染快速 | MuseScore (GPL 隔离) | 🧪 v0.3 |

## 许可证

主项目：**MIT**（参见 [LICENSE](LICENSE)）。

GPL 子项目（`render-lilypond/`、`render-musescore/`）进程隔离，详见 [design.md §8](docs/design.md#8-依赖分层)。

## 开发环境

- Python ≥ 3.11
- 包管理：[uv](https://github.com/astral-sh/uv)（`brew install uv`）
- GPU 推理：PyTorch + CUDA
- 容器化：Docker + NVIDIA Container Toolkit
- 预提交：ruff + mypy + 许可证扫描

## 快速开始

```bash
# 安装 uv（首次）
brew install uv

# 锁定 Python 版本
uv python install 3.11
uv python pin 3.11

# 创建 venv 并装依赖
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 跑测试
pytest tests/

# 跑最小管线（首次会下载模型权重到 ~/.cache/huggingface/）
mujik run --input song.wav --output out/ --config config/default.yaml
```

## 仓库结构

```
mujik-transcriptor/
├── pyproject.toml
├── LICENSE                      # MIT
├── Dockerfile
├── docker-compose.yml
├── docs/
│   ├── research.md
│   ├── design.md
│   └── ...
├── config/
├── src/mujik/
│   ├── pipeline.py
│   ├── preprocess/
│   ├── separate/
│   ├── transcribe/
│   ├── rhythm/
│   ├── chord/
│   ├── midi/
│   ├── quantize/
│   ├── merge/
│   ├── time_signature/
│   ├── score/
│   └── render/
├── render-lilypond/             # GPL-2.0+ 独立子项目
├── render-musescore/            # GPL-2.0+ 独立子项目
├── tests/
├── benchmarks/
└── weights/                     # .gitignore
```

## 贡献

WIP。本地 benchmark 是当前最重要的环节：见 [research.md §6](docs/research.md#6-本地-benchmark-清单)。

## 引用

引用本项目时，请同时引用上游模型：

- Demucs v4：Defossez et al., 2022, [arXiv:2211.08553](https://arxiv.org/abs/2211.08553)
- Beat Transformer：Zhao et al., ISMIR 2022, [arXiv:2209.07140](https://arxiv.org/abs/2209.07140)
- BeatNet：Heydari et al., [arXiv:2108.03576](https://arxiv.org/abs/2108.03576)
- basic-pitch：[Spotify Research](https://github.com/spotify/basic-pitch)
- adtof / ADTOF dataset：Zehren et al., 2021, [arXiv:2107.05535](https://arxiv.org/abs/2107.05535)

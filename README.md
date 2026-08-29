# mujik-transcriptor

> 端到端音乐音频 → MIDI + PDF 乐谱处理管线
> 目标：流行/摇滚/民谣 + 爵士/复杂和声 + 金属/复杂节奏
> 全开源 / GPU 离线 / 保留演奏细节

## 状态

✅ **v0.5.0（功能完整）** — 17/17 E2E smoke phases passing · 681 unit tests · 0-0 mainline

| Stream | 最近版本 | 状态 |
|---|---|---|
| 源分离 (Demucs v4 4-stem / 6-stem / Roformer) | v0.4.0 | ✅ |
| 转录 (basic-pitch / adtof / bytedance-piano / multitrack) | v0.4.2 | ✅ |
| 节拍/下拍 (madmom CRNN) | v0.2.2 | ✅ |
| 时间签名（启发式 4/4 + change CLI）| v0.2.3 | ✅ |
| 自动和弦（madmom + BTC-HCQT 延伸）| v0.4.8 | ✅ |
| Chord 后处理（quantize / groove / hardening）| v0.4.9 | ✅ |
| MusicXML 渲染（bend / harmony）| v0.4.3 | ✅ |
| 5-genre benchmark 框架 | v0.5.0 | ✅ |

## 🚀 5 分钟跑通

最快路径：**Docker 镜像**（零本地依赖）。

```bash
# 1. 拉镜像（包含 demucs / madmom / verovio 全部预装）
docker pull ghcr.io/alubawon/mujik-transcriptor:dev-v0.5.1

# 2. 跑 demo（输入你手头的 wav，30s 内出 MIDI + PDF）
docker run --rm -v $(pwd):/work -w /work \
  ghcr.io/alubawon/mujik-transcriptor:dev-v0.5.1 \
  mujik run --input song.wav --output out/ --preset pop

# 3. 看产物
ls out/
# project.mid  score.pdf  chords.json  beats.json  time_signatures.json  project.json
```

> GPU 加速：把 `--device cuda` 加到 docker run（需 NVIDIA Container Toolkit）。

## 🪄 一键 demo（需要真实音频）

```bash
# 跑 pop/jazz/metal 三 preset 对比 → demo_out/{pop,jazz,metal}/ + demo_report.md
./scripts/run_demo.sh path/to/your_song.wav

# 可选：自定义裁剪时长（需要 ffmpeg）
./scripts/run_demo.sh path/to/your_song.wav 30
```

> **必须提供真实 wav**（pop/jazz/metal 三个 preset 差异只在真实音乐上才可见；
> 跑合成正弦波对所有 preset 输出一致，没有 demo 价值）。支持 `.wav` / `.flac` / `.mp3` / `.ogg` / `.m4a`。
>
> 不接受任何默认值：脚本会显式报错并打印用法。
```

输出 `out/pop/`, `out/jazz/`, `out/metal/`，每个含 MIDI + PDF + JSON 报告。

## 📖 调用示例

| 任务 | 命令 |
|---|---|
| 完整管线（流行 preset）| `mujik run --input song.wav --output out/ --preset pop` |
| 仅分轨 | `mujik separate --input song.wav --output stems/` |
| 量化 MIDI 到 grid | `mujik quantize --project-dir out/ --out-dir out_q/` |
| 单独跑和弦检测 | `mujik chords --input song.wav --output chords.json` |
| 多乐器一次转（MuScriptor）| `mujik multitrack --input song.wav --output out_mt/ --model small` |
| MusicXML → PDF | `mujik render --input score.musicxml --output score.pdf --pdf` |
| 5-genre benchmark | `python -m mujik.benchmarks.runner --dataset synthetic` |
| 用 YAML 自定义配置 | `mujik run --input song.wav --output out/ --config my.yaml` |

详见 [调用示例 §Examples](#examples) 段落。

## 架构一览

```
WAV/FLAC/MP3
  → pyloudnorm (响度归一)
  → Demucs v4 htdemucs_ft (4-stem 源分离)
  → basic-pitch / adtof / bytedance / multitrack (各轨 MIDI 转录)
  → madmom CRNN (节拍/下拍) + 启发式 (时间签名)
  → madmom / BTC-HCQT (和弦识别，opt-in)
  → quantize + chord quantize + chord groove (后处理)
  → mido + pretty-midi + music21 (MIDI 操作)
  → Verovio (BSD) / LilyPond 隔离 / MuseScore 隔离 (乐谱渲染)
  → MIDI + PDF
```

详见 [设计文档 §2](docs/design.md#2-架构总览)。

## 当前选型基线

| 环节 | 首选 | 状态 |
|---|---|---|
| 源分离 | Demucs v4 htdemucs_ft | ⭐ v0.4.0 6-stem |
| 旋律/贝斯/other 转录 | basic-pitch | ✅ v0.2.1 |
| 鼓转录 | adtof | ✅ v0.2.1 |
| 钢琴转录 | bytedance-piano | ✅ v0.4.0 |
| 节拍/下拍 | madmom CRNN | ✅ v0.2.2 |
| 时间签名 | 启发式 4/4 | ✅ v0.2.3 |
| 和弦识别（major/minor）| madmom CRNN | ✅ v0.4.4 |
| 和弦识别（7th/延伸）| BTC-HCQT (MIT) | ✅ v0.4.8 |
| 多乐器一次转 | MuScriptor（CC-BY-NC 隔离）| ✅ v0.4.2 |
| 渲染主线 | Verovio (BSD) | ✅ v0.2.4 |
| 渲染精细 | LilyPond (GPL 隔离) | 🧪 v0.3 |
| 渲染快速 | MuseScore (GPL 隔离) | 🧪 v0.3 |
| Benchmark 框架 | synthetic 5-genre | ✅ v0.5.0 |

## 文档

- [docs/research.md](docs/research.md) — 选型研究报告
- [docs/design.md](docs/design.md) — 设计文档
- [scripts/run_demo.sh](scripts/run_demo.sh) — 一键 demo 脚本
- [config/default.yaml](config/default.yaml) — 默认配置
- [config/presets/](config/presets/) — pop/jazz/metal 三 preset

## <a id="examples"></a>📋 完整调用示例

### 1. 全管线 — 流行 preset
```bash
mujik run --input song.wav --output out/ --preset pop
# out/project.mid   (MIDI)
# out/score.musicxml  (MusicXML 3.1 with <harmony>)
# out/score.pdf     (Verovio 渲染)
# out/chords.json   (ChordEvent 列表)
# out/beats.json + time_signatures.json
# out/project.json  (metadata sidecar)
```

### 2. 爵士 preset（开 chord + 延伸和弦 + quantize）
```bash
mujik run --input jazz_take.wav --output out_jazz/ --preset jazz
# jazz preset 默认开启 chord + chord.quantize，gpt-2.7B chord quality
```

### 3. 金属 preset（开 6-stem Demucs + ByteDance piano）
```bash
mujik run --input metal_track.wav --output out_metal/ --preset metal
# metal preset 用 htdemucs_6s，多出 piano/guitar 两轨
```

### 4. 多乐器一次转（MuScriptor，替代 4-stem）
```bash
export HF_TOKEN=hf_xxx  # muscriptor 权重 CC-BY-NC 4.0
mujik multitrack --input song.wav --output out_mt/ --model small
```

### 5. 单独和弦检测
```bash
mujik chords --input song.wav --output chords.json
# 仅跑 madmom CRNN chord，5-30s 出 major/minor ChordEvent
```

### 6. 量化 MIDI 到 grid
```bash
mujik quantize --project-dir out/ --out-dir out_q/
# 在 out_q/ 生成 project.mid + quantize_report.json
```

### 7. 改拍号
```bash
mujik time-signature change --project-dir out/ --at 1:30.000 \
  --to 3/4 --mode A
```

### 8. MusicXML → PDF
```bash
mujik render --input score.musicxml --output score.pdf --pdf
# backend: verovio (默认) / lilypond / musescore
```

### 9. 5-genre benchmark
```bash
uv pip install 'mujik-transcriptor[benchmark]'
python -m mujik.benchmarks.runner --dataset synthetic --output bench.md
# 输出 bench.md + bench.json（per-genre + overall）
```

## 许可证

主项目：**MIT**（参见 [LICENSE](LICENSE)）。

GPL 子项目（`render-lilypond/`、`render-musescore/`）进程隔离，详见 [design.md §8](docs/design.md#8-依赖分层)。

muscriptor 适配器（v0.4.2+）通过 **subprocess 隔离** 调 `uvx muscriptor`，主线不 import muscriptor 包。
muscriptor 模型权重为 **CC BY-NC 4.0**（仅限非商用研究），项目本身已声明非商用前提。详见 [docs/MUSCRIPTOR_INTEGRATION.md](docs/MUSCRIPTOR_INTEGRATION.md#许可证声明)。

## 开发环境

- Python ≥ 3.11
- 包管理：[uv](https://github.com/astral-sh/uv)（`brew install uv`）
- GPU 推理：PyTorch + CUDA
- 容器化：Docker + NVIDIA Container Toolkit
- 预提交：ruff + mypy + 许可证扫描

### 本地开发 setup
```bash
# 锁定 Python 版本
uv python install 3.11
uv python pin 3.11

# 创建 venv 并装依赖
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,benchmark]"

# 跑测试
pytest tests/ -q

# 跑最小管线（首次会下载模型权重到 ~/.cache/huggingface/）
mujik run --input song.wav --output out/ --config config/default.yaml

# 跑一键 demo（需要真实 wav）
./scripts/run_demo.sh path/to/your_song.wav
```

## 仓库结构

```
mujik-transcriptor/
├── pyproject.toml
├── LICENSE                      # MIT
├── Dockerfile
├── docker-compose.yml
├── scripts/
│   └── run_demo.sh              # 一键 demo（v0.5.1）
├── docs/
│   ├── research.md
│   ├── design.md
│   └── ...
├── config/
│   ├── default.yaml
│   └── presets/                 # pop/jazz/metal
├── src/mujik/
│   ├── pipeline.py
│   ├── cli.py
│   ├── preprocess/
│   ├── separate/
│   ├── transcribe/
│   ├── rhythm/
│   ├── chord/
│   ├── midi/
│   ├── quantize/
│   ├── benchmarks/              # v0.5.0
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
- madmom CRNN：[CPJKU/madmom](https://github.com/CPJKU/madmom)
- BTC-ISMIR19：Park et al., 2019 ([paper](https://program.ismir.net/2019/ISMIR_53.html))
- MuScriptor：Kyutai + Mirelo, [muscriptor](https://github.com/muscriptor/muscriptor)
- basic-pitch：[Spotify Research](https://github.com/spotify/basic-pitch)
- adtof / ADTOF dataset：Zehren et al., 2021, [arXiv:2107.05535](https://arxiv.org/abs/2107.05535)

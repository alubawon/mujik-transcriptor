# mujik-transcriptor 设计文档

> 版本：v0.1（2026-08-26）
> 状态：基于 [research.md](./research.md) 的研究结论 + 4 条用户澄清

## 1. 项目目标

`mujik-transcriptor` 是一个端到端音乐音频处理管线，输入任意流行/爵士/金属/电子音乐，输出 MIDI + 可选 PDF 乐谱。

**核心场景**：
- 个人/研究自用
- 离线批处理 + GPU 加速
- 流行/摇滚/民谣 + 爵士/复杂和声 + 金属/复杂节奏
- 保留演奏细节（力度 / 弯音 / 滑音）
- 多轨合板为钢琴缩谱或总谱
- 严格节拍量化

## 2. 架构总览

### 2.1 进程边界

```
                ┌─────────────────────────────────────────────┐
                │      主线：mujik-core（MIT, PyTorch）        │
                │                                              │
   audio in ──▶ │  preprocess ─▶ separate ─▶ transcribe ─┐   │
                │  (pyloudnorm)   (Demucs)   (basic-pitch)│   │
                │                              (adtof)    │   │
                │                                         ▼   │
                │                                    rhythm    │
                │                                    (Beat     │
                │                                    Transformer│
                │                                    /BeatNet) │
                │                                         │   │
                │                                         ▼   │
                │                                    quantize  │
                │                                    + merge   │
                │                                         │   │
                │                                         ▼   │
                │                                    score     │
                │                                    (MusicXML)│
                │                                         │   │
                │            ┌──── Verovio (BSD, in-proc) ──┐│
                │            │                              ││
                │            ▼                              ││
                │       PDF/SVG ─────────────────────────┐  ││
                │                                       │  ││
                │            ┌─── HTTP (GPL 隔离) ────┐  │  ││
                │            │                        │  │  ││
                │            ▼                        ▼  ▼  ▼│
                │       render-lilypond          render-musescore
                │       (GPL-2.0+)              (GPL-2.0+)
                └─────────────────────────────────────────────┘
```

### 2.2 模块清单

| 模块 | 路径 | 职责 | 关键依赖 |
|---|---|---|---|
| `preprocess` | `src/mujik/preprocess/` | 响度归一、去混响、去噪 | pyloudnorm, deEchoes, nnnoiseless |
| `separate` | `src/mujik/separate/` | 源分离（4/5/6-stem 可插拔） | Demucs v4 / MDX23C / Roformer |
| `transcribe` | `src/mujik/transcribe/` | MIDI 转录（按 stem 路由） | basic-pitch, adtof, ByteDance piano |
| `rhythm` | `src/mujik/rhythm/` | 节拍/下拍/时间签名 | Beat Transformer, BeatNet, ResNet18 |
| `chord` | `src/mujik/chord/` | 和弦识别 | BTC-HCQT, CREMA, Chord-CNN-LSTM |
| `midi` | `src/mujik/midi/` | MIDI 读写、事件编辑 | pretty-midi, mido, music21 |
| `quantize` | `src/mujik/quantize/` | 网格量化、groove 模板 | mido + 自研 |
| `merge` | `src/mujik/merge/` | 多轨合并三档 | music21 |
| `time_signature` | `src/mujik/time_signature/` | 分段拍号数据模型 + 改拍号两种模式 | 标准库 |
| `score` | `src/mujik/score/` | MusicXML 生成 | music21 |
| `render` | `src/mujik/render/` | Verovio（BSD 内嵌）+ GPL 渲染客户端 | verovio, httpx |

## 3. 数据模型

### 3.1 TimeSignatureSegment

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class TimeSignatureSegment:
    """时间签名的一段。模型把"分段拍号"作为一等公民，支持变拍子。"""
    start_time: float          # 秒（开区间起点）
    end_time: float            # 秒（开区间终点）
    time_signature: tuple[int, int]  # (numerator, denominator)，如 (4, 4)
    confidence: float          # 0-1，自动识别置信度
    source: Literal[
        "auto_resnet18",       # ResNet18/METER2800 自动
        "auto_beatnet",        # BeatNet 自动
        "manual",              # 用户手动
        "default_4_4",         # 默认兜底
    ]
```

### 3.2 Note

```python
@dataclass(frozen=True)
class Note:
    """MIDI 事件的最小单位。绝对时间戳，不依赖拍号。"""
    start: float         # 秒
    end: float           # 秒
    pitch: int           # 0-127
    velocity: int        # 0-127
    channel: int = 0     # 0-15
    pitch_bend: tuple[float, ...] = ()  # 帧级弯音序列

    def duration(self) -> float:
        return self.end - self.start
```

### 3.3 Stems

```python
from typing import Literal
from pathlib import Path

StemName = Literal[
    "vocals", "drums", "bass", "other",
    "piano", "guitar",  # 仅 5/6-stem 模式
]

@dataclass
class Stem:
    name: StemName
    audio_path: Path
    sample_rate: int
    duration: float
    source_model: str

@dataclass
class Stems:
    """一次源分离产出的所有 stem。4/5/6+ stem 通过此容器统一管理。"""
    stems: dict[StemName, Stem]
    stem_count: int
    separation_model: str
    separation_time: float
```

### 3.4 Track / Project

```python
@dataclass
class Track:
    stem_name: StemName
    notes: list[Note]
    instrument: str
    channel: int

@dataclass
class Project:
    audio_path: Path
    duration: float
    sample_rate: int
    time_signatures: list[TimeSignatureSegment]
    tempo_map: list[TempoSegment]
    tracks: dict[StemName, Track]
    chord_track: list[ChordEvent] | None
    metadata: dict
```

## 4. 改拍号两种模式

### 4.1 模式 A：按现有时间轴在新拍号下堆积

**语义**：note 的绝对时间戳不变；按新拍号重画小节线。

**适用**：用户改错了自动识别（如 3/4 被误识别为 4/4）。

**实现**：

```python
def redraw_bars_under_new_time_signature(
    notes: list[Note],
    old_segments: list[TimeSignatureSegment],
    new_signature: tuple[int, int],
    apply_range: tuple[float, float],
) -> tuple[list[Note], list[TimeSignatureSegment]]:
    """模式 A：保留 note 时间戳，重画小节线。"""
    # 1. 在 apply_range 内生成新拍号的小节边界序列
    # 2. note 不修改
    # 3. 更新 TimeSignatureSegment 序列
    ...
```

### 4.2 模式 B：按当前小节改拍号后填充/阶段

**语义**：变更点之前的已填小节保持原样；变更点之后按新拍号重排。

**子模式**：
- `B1.preserve_time`：保留 note 相对时间戳，仅重画小节线（类似模式 A 但只在变更点后）
- `B2.regrid`：按新拍号 grid 重排 note（在变更点后 note 重排到最近 grid 点）

**适用**：曲式真的从 4/4 过渡到 7/8。

**实现**：

```python
def change_time_signature_at_boundary(
    notes: list[Note],
    segments: list[TimeSignatureSegment],
    change_time: float,
    new_signature: tuple[int, int],
    mode: Literal["preserve_time", "regrid"],
) -> tuple[list[Note], list[TimeSignatureSegment]]:
    """模式 B：在 change_time 处分段。"""
    # 1. 分割旧的 TimeSignatureSegment
    # 2. 在 change_time 后插入新拍号段
    # 3. 根据 mode 处理 note
    ...
```

## 5. 配置 schema（pydantic）

```python
# src/mujik/config/schema.py
from pydantic import BaseModel, Field
from typing import Literal

class SourceSeparationConfig(BaseModel):
    stem_count: Literal[4, 5, 6] = 4
    model: Literal["demucs", "mdx23c", "bsroformer", "melbandroformer"] = "demucs"
    variant: str = "htdemucs_ft"
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    precision: Literal["fp32", "fp16", "bf16"] = "fp16"
    segment_length: float = 7.5
    overlap: float = 0.25

class TranscribeConfig(BaseModel):
    vocals: str = "basic-pitch"
    bass: str = "basic-pitch"
    drums: str = "adtof"
    piano: str = "byteance-piano"
    guitar: str = "apollo"
    other: str = "basic-pitch"
    polyphonic_threshold: float = 0.5
    onset_interval_min_ms: float = 50.0
    velocity_threshold: int = 30

class RhythmConfig(BaseModel):
    beat_tracker: Literal["beat-transformer", "beatnet"] = "beat-transformer"
    time_signature_model: str = "resnet18-meter2800"
    time_signature_fallback: tuple[int, int] = (4, 4)
    allow_user_override: bool = True

class QuantizeConfig(BaseModel):
    enabled: bool = True
    grid_resolution: int = 16
    strength: float = 0.8
    groove_template: str = "straight"

class MergeConfig(BaseModel):
    mode: Literal["all", "piano_reduction", "score"] = "piano_reduction"
    density_filter: bool = True
    max_simultaneous_notes: int = 12

class RenderConfig(BaseModel):
    pdf_backend: Literal["verovio", "lilypond", "musescore"] = "verovio"
    lilypond_url: str = "http://localhost:5001"
    musescore_url: str = "http://localhost:5002"
    include_chord_symbols: bool = True
    include_lyrics: bool = False
    page_size: Literal["A4", "Letter"] = "A4"
    staff_count: int = 2

class PipelineConfig(BaseModel):
    input_path: str
    output_dir: str
    preset: Literal["pop", "jazz", "metal", "custom"] = "custom"
    source_separation: SourceSeparationConfig = Field(default_factory=SourceSeparationConfig)
    transcribe: TranscribeConfig = Field(default_factory=TranscribeConfig)
    rhythm: RhythmConfig = Field(default_factory=RhythmConfig)
    quantize: QuantizeConfig = Field(default_factory=QuantizeConfig)
    merge: MergeConfig = Field(default_factory=MergeConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
```

## 6. API 契约

### 6.1 主线 Python API

```python
from mujik import Pipeline, PipelineConfig

config = PipelineConfig.from_yaml("config/default.yaml")
pipeline = Pipeline(config)

project = pipeline.run(
    input_path="song.wav",
    output_dir="out/",
)

# 产物：
# out/
# ├── project.mid            # 合并后的主 MIDI
# ├── stems/                 # 分离后音频
# │   ├── vocals.wav
# │   ├── drums.wav
# │   ├── bass.wav
# │   └── other.wav
# ├── tracks/                # 每 stem 独立 MIDI
# ├── beats.json
# ├── chords.json
# ├── score.musicxml
# └── score.pdf
```

### 6.2 渲染服务 HTTP 接口

```python
# render-lilypond/server.py (FastAPI)
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class RenderRequest(BaseModel):
    input_type: Literal["musicxml", "midi"]
    input_b64: str
    options: dict

class RenderResponse(BaseModel):
    pdf_b64: str
    musicxml_out: str | None = None

@app.post("/render")
def render(req: RenderRequest) -> RenderResponse:
    # LilyPond CLI 调用
    ...
```

## 7. 仓库结构

```
mujik-transcriptor/
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE                       # MIT
├── Dockerfile                    # 主线 GPU 镜像
├── Dockerfile.cpu                # 主线 CPU 镜像
├── docker-compose.yml
├── .pre-commit-config.yaml
├── .gitignore
├── docs/
│   ├── research.md
│   ├── design.md
│   ├── time-signature-modes.md
│   ├── license.md
│   └── benchmarks.md
├── config/
│   ├── default.yaml
│   ├── schemas/
│   └── presets/
├── src/mujik/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── config/
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
├── render-lilypond/              # GPL-2.0+
├── render-musescore/             # GPL-2.0+
├── tests/
├── benchmarks/
└── weights/                      # .gitignore
```

## 8. 依赖分层

主项目依赖**仅限 MIT/Apache/BSD**：

| 类别 | 包 | 许可证 |
|---|---|---|
| 核心 | numpy, scipy, pydantic, pyyaml, loguru | BSD/MIT |
| 音频 I/O | soundfile, pyloudnorm | BSD/MIT |
| MIDI | pretty-midi, mido, music21 | MIT/BSD-3 |
| 源分离 | demucs | MIT |
| 转录（PyTorch） | torch, torchaudio, onnxruntime | BSD/Apache |
| 渲染（主线） | verovio | BSD-3 |
| HTTP 客户端 | httpx | BSD |

**GPL 隔离**（独立子项目）：
- `render-lilypond/` — GPL-2.0+，FastAPI + LilyPond CLI
- `render-musescore/` — GPL-2.0+，FastAPI + MuseScore CLI

**避开**：
- Facebook denoiser（CC-BY-NC-4.0）→ 自研或 nnnoiseless
- Essentia（AGPL-3.0）→ 自研
- Chordino（GPL-3.0）→ BTC-HCQT / CREMA

**basic-pitch 的 TF 依赖**：通过 `optional-dependencies.transcribe-tf` 隔离，主线在 `transcribe.basic_pitch` 模块用 subprocess 调用。

## 9. 风险登记

| # | 风险 | 等级 | 缓解 |
|---|---|---|---|
| 1 | 复调转录谐波"幻觉" | 高 | 阈值过滤 + ByteDance piano for piano 轨 |
| 2 | 金属鼓 blast beat 鲁棒性 | 高 | adtof vs Omnizart 本地 F1 + 微调 |
| 3 | 罕见拍号自动识别失败 | 高 | 分段拍号数据模型 + 改拍号两种模式 + UI 覆盖 |
| 4 | 多轨合并复调密度爆炸 | 中 | 三档输出 + 密度过滤 |
| 5 | 演奏细节（弯音/滑音/踏板）丢失 | 中 | CREPE/pyworld f0 → pitch bend |
| 6 | groove 模板无现成开源 | 中 | 自实现偏移查找表 |
| 7 | **GPL 传染** | 高 | 进程隔离 + 中间格式 + 仓库分层 |
| 8 | **基础栈混合 TF + PyTorch** | 中 | TF 走子进程，主线 PyTorch |
| 9 | **依赖版本漂移** | 高 | uv.lock + Docker 镜像固定 |
| 10 | **模型权重不入仓的存储管理** | 中 | HF Hub + DVC + 共享卷 |
| 11 | 时间签名仅 4 类 | 高 | 已在 #3 处理 |
| 12 | 6-stem 模式 GPU 内存爆炸 | 中 | 段长控制 + fp16 |

## 10. 路线图

### v0.1（地基，1-2 周）
- [ ] 仓库结构 + 依赖锁定
- [ ] 数据模型 + 单测
- [ ] Demucs v4 htdemucs_ft 4-stem 源分离
- [ ] Verovio PDF 渲染（BSD，先不依赖 GPL）
- [ ] 端到端最小路径：WAV → stems → MIDI → MusicXML → PDF
- [ ] pre-commit 钩子（ruff + mypy + 许可证扫描）

### v0.2（核心管线，2-3 周）
- [ ] basic-pitch 转录（vocals/bass/other）
- [ ] adtof 转录（drums）
- [ ] Beat Transformer 节拍/下拍
- [ ] ResNet18/METER2800 时间签名（4 类）
- [ ] mido 量化
- [ ] 改拍号两种模式（CLI）
- [ ] Verovio 出 PDF（完整）

### v0.3（GPL 渲染隔离，1 周）
- [ ] `render-lilypond/` 子项目独立仓库（GPL-2.0+）
- [ ] `render-musescore/` 子项目独立仓库（GPL-2.0+）
- [ ] 主线 `render-client` HTTP 封装
- [ ] 三档输出（全合 / 钢琴缩谱 / 总谱）

### v0.4（扩展，2 周）
- [ ] 5/6-stem 可插拔（MDX23C / BS-Roformer）
- [ ] ByteDance piano for piano stem
- [ ] Chord-CNN-LSTM
- [ ] 弯音/滑音注入子管线
- [ ] swing/groove 模板自实现
- [ ] 改拍号两种模式 GUI

### v0.5（验收，1-2 周）
- [ ] 5 类音乐各 5-10 首基准曲库
- [ ] SDR/SIR/SAR、F1、Chord acc 全指标
- [ ] 文档化决策

## 11. 开放问题（用户已确认）

1. **主项目许可证 MIT** ✅
2. **基本栈统一到 PyTorch**（basic-pitch 的 TF 通过子进程隔离） ✅
3. **5/6-stem 何时启用**：v0.1 先 4-stem ✅
4. **是否需要 GUI / 服务化**：后续追加 ✅
5. **PDF 优先 LilyPond（精细）**：v0.1 用 Verovio 占位，v0.3 切到 LilyPond ✅

---

> 对应研究依据见 [research.md](./research.md)


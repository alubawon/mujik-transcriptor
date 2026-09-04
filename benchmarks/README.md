# benchmarks/

Benchmark scripts + the v0.5.0 5-genre benchmark framework (`src/mujik/benchmarks/`).

## Framework（src/mujik/benchmarks/，v0.5.0+）

- `datasets/synthetic.py` — 内置 5 genre × 3 file synthetic baseline（CI smoke，无外部数据）
- `datasets/local.py` — **v0.5.2**：manifest 驱动的本地真实曲库（仓库不携带数据，版权干净）
- `pipeline_adapter.py` — **v0.5.2**：把完整管线（demucs + madmom + basic-pitch/drumscript + chord）接进 runner
- `metrics.py` — note F1（±50ms onset）/ beat CMLt / chord majmin（mir_eval）
- `runner.py` — 编排 + 聚合 + CLI 入口

### 真实数据 benchmark（--dataset local）

仓库不携带任何音频/标注。在自家曲库目录放 `manifest.json`：

```json
[
  {
    "sample_id": "s1",
    "genre": "jazz",
    "audio": "audio/s1.wav",
    "notes":  [[60, 0.5, 1.2]],
    "beats":  [0.0, 0.5, 1.0],
    "chords": [[0.0, 2.0, "C", "maj7"]]
  }
]
```

`notes`/`beats`/`chords` 均可选（缺失的 metric 该样本记 0）。
路径相对 data_dir；manifest 或音频缺失/格式错 → fail-loud。

```bash
.venv/bin/python -m mujik.benchmarks.runner \
  --dataset local --data-dir my_bench/ \
  --preset pop --output bench.md --json bench.json
```

对每个样本跑一次完整管线（`--preset pop/jazz/metal`，chord 默认强制开，
`--no-chords` 关闭）。`--limit N` 只跑前 N 个样本。

## Scripts（v0.4.0 scaffold）

### `run_separation.py`
Run Demucs (4-stem `htdemucs_ft` or 6-stem `htdemucs_6s`) and report timing + volume.

```bash
PYTHONPATH=src python benchmarks/run_separation.py \
    --input /path/to/song.wav \
    --variant htdemucs_6s \
    --device cuda \
    --out /tmp/separation_report.json
```

### `run_transcription.py`
Run a transcription adapter (basic-pitch / drumscript / bytedance-piano) and report timing + note count.

```bash
PYTHONPATH=src python benchmarks/run_transcription.py \
    --input /path/to/song.wav \
    --adapter basic-pitch \
    --out /tmp/transcription_report.json
```

## Roadmap

- ✅ ~~v0.5: framework + synthetic baseline~~（v0.5.0 落地）
- ✅ ~~v0.5.2: 真实数据 benchmark~~（`--dataset local` manifest 驱动）
- ✅ ~~v0.5.2: 分离质量 benchmark~~（`separation.py`，MUSDB18 + museval SDR/SIR/SAR，见下）
- 标准公开数据集（MAPS 钢琴 / Billboard 和弦）按 PR 引入
- 评测项对照 [research.md §6 本地 benchmark 清单](../docs/research.md#6-本地-benchmark-清单)
  （优先级 1+2+3+6：分离 SDR / 多音转录 F1 / 鼓转录 / 演奏细节 f0-vs-MIDI）

## Separation benchmark（MUSDB18 + museval，v0.5.2）

仓库不携带 MUSDB18（research-only 许可），先自行下载
[MUSDB18 / MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html)：

```bash
uv pip install 'mujik-transcriptor[separation-bench]'

# MUSDB18-HQ（wav，推荐，无需 ffmpeg）
.venv/bin/python -m mujik.benchmarks.separation \
  --musdb-root ~/data/musdb18-hq --is-wav \
  --variant htdemucs_ft --device cpu --limit 3 \
  --output sep_bench.md --json sep_bench.json

# 压缩版 MUSDB18（.mp4 stems，解码需 ffmpeg）：去掉 --is-wav
```

输出 per-stem（vocals/drums/bass/other）median SDR/SIR/SAR + per-track 明细。
分离复用 `mujik.separate.router.separate_audio`（variant 可换 htdemucs_6s）。

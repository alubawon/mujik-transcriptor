# MuScriptor Integration (v0.4.2)

## 概述

v0.4.2 起，mujik-transcriptor 支持通过 [MuScriptor](https://github.com/muscriptor/muscriptor)（Kyutai + Mirelo 联合出品）进行**多乐器一次性转录**，作为 4/5/6-stem 分轨转录的替代方案。

## 选型理由

| 同定位多乐器转录模块 | ★ | License | 状态 | 评价 |
|---|---|---|---|---|
| `magenta/mt3` | 1744 | Apache-2.0 | 2026-07 活跃 | 多乐器标杆，但 T5X + JAX 集成重 |
| **`muscriptor`** | 1275 | 代码 MIT / 权重 CC-BY-NC | 2026-08-24 (4 天前) | **黑马**；PyPI 即装即用 |
| `gudgud96/MR-MT3` | 57 | MIT | 2025-06 | MT3 改进版 |
| `sony/hFT-Transformer` | 120 | MIT | 2023-07 停更 | 老但成熟 |
| `Apollo` (Sony 推测) | 0 | — | **未公开** | 4 个候选 URL 全 404 |

v0.4.2 选 muscriptor 的原因：
- ✅ 代码 MIT（`liccheck` 合规）
- ✅ PyPI 直接 `pip install muscriptor`（v0.3.0，2026-08-05）
- ✅ 输出多乐器多轨 MIDI（vocals/drums/bass/piano/guitar）
- ✅ 4 天前刚更新（最活跃）
- ⚠️ 模型权重 **CC-BY-NC 4.0**（非商用）— 项目本身已声明非商用，进程隔离为独立 CLI 调用，**主线不 import muscriptor 包** → 不触发 `liccheck` 警告

## 安装

```bash
# 主线 (mujik-transcriptor v0.4.2+)
pip install mujik-transcriptor

# muscriptor multitrack adapter (optional extra)
pip install 'mujik-transcriptor[transcribe-muscriptor]'
# 或独立装 muscriptor 包
pip install muscriptor
```

muscriptor 通过 `uvx` 调用（推荐方式），先装 uv：

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或
pip install uv
```

## HuggingFace Token 配置

muscriptor 模型权重 CC-BY-NC 4.0，**必须**在 HuggingFace 接受 license 后才能下载。

1. 访问 https://huggingface.co/MuScriptor/muscriptor-small（或 medium / large）
2. 点击 "Agree and access repository" 接受 CC BY-NC 4.0 license
3. 创建 token: https://huggingface.co/settings/tokens
4. 设置环境变量：

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

## CLI 用法

### multitrack 独立子命令

```bash
# 默认 medium 模型
mujik multitrack --input song.wav --output out_dir/

# 小模型（CPU 友好，103M 参数）
mujik multitrack --input song.wav --output out_dir/ --model small

# 大模型（GPU 推荐，1.4B 参数）
mujik multitrack --input song.wav --output out_dir/ --model large

# 自定义超时（默认 1800s = 30 分钟）
mujik multitrack --input song.wav --output out_dir/ --timeout 3600
```

### run 管线（multitrack 模式）

通过 YAML 配置或 `--config` 切换转录模式：

```yaml
# config_multitrack.yaml
transcribe:
  mode: multitrack
  muscriptor_model: medium
```

```bash
mujik run --input song.wav --output out_dir/ --config config_multitrack.yaml
```

multitrack 模式会**跳过** 4/5/6-stem 源分离（Demucs），直接对完整音频做多乐器转录。

## 输出

`mujik multitrack` 在 `out_dir/` 产生：
- `project.mid` — 多轨 MIDI（含 vocals/drums/bass/piano/guitar 多个 instrument）
- `project.json` — 元数据（muscriptor_model、tracks、total_notes）
- 后续 `mujik render` 仍可正常生成 SVG / PDF 乐谱

muscriptor 默认 MIDI 输出位置：`<out_dir>/<audio_stem>.mid`

## 模型尺寸选择

| 尺寸 | 参数 | 推荐硬件 | 速度 | 准确度 |
|---|---|---|---|---|
| `small` | 103M | CPU | 快 | 一般 |
| `medium`（默认） | 307M | GPU 推荐 / Apple Silicon | 中 | 较好 |
| `large` | 1.4B | GPU 必需 | 慢 | 最佳 |

## 平台支持

- Linux：✅ CPU / CUDA / MPS（如有）
- macOS Apple Silicon：✅ MPS 自动
- macOS Intel：⚠️ 需 Python ≤ 3.12（PyTorch 2.2.2 后停止 x86_64 wheels）
- Windows：CPU 默认；GPU 需 `uvx --torch-backend=cu128 muscriptor ...`

## 故障排查

### `uvx not found`
```bash
pip install uv
# 或 https://docs.astral.sh/uv/getting-started/installation/
```

### `HuggingFace authentication failed`
1. 确认 `HF_TOKEN` 已设置：`echo $HF_TOKEN`
2. 确认已访问 https://huggingface.co/MuScriptor/muscriptor-{small,medium,large} 接受 license
3. 确认 token 有 `read` 权限

### `CUDA out of memory`
```bash
# 换小模型
mujik multitrack --input song.wav --output out_dir/ --model small

# 或缩短音频
ffmpeg -i long_song.wav -t 60 short_song.wav
mujik multitrack --input short_song.wav --output out_dir/
```

### muscriptor timeout
```bash
# 加超时
mujik multitrack --input song.wav --output out_dir/ --timeout 3600
```

## 许可证声明

- **muscriptor 代码**：[MIT License](https://github.com/muscriptor/muscriptor/blob/main/LICENSE)
- **muscriptor 模型权重**：[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)（**仅限非商业研究**）
- mujik-transcriptor 本项目：MIT License

mujik-transcriptor 本身以非商用为前提（在 v0.4.0+ LICENSE addendum 中声明），muscriptor 权重 CC-BY-NC 4.0 限制可接受。**商业使用** muscriptor 需联系 Kyutai / Mirelo 获取商用许可，或在 mujik-transcriptor 中换用 Apache-2.0 的同定位模型（如 `magenta/mt3`，但 T5X + JAX 集成重，留 v0.5+ 评估）。

## 不在 v0.4.2 范围

- ❌ muscriptor sheets format（MuseScore 4+ GPL 进程隔离，留 v0.4.3+）
- ❌ muscriptor 量化 MIDI 自动选择（v0.4.3+）
- ❌ HF login UI（仅 env var）
- ❌ MT3 / MR-MT3 集成（v0.4.3+ 评估）

## 关键参考

- muscriptor PyPI: https://pypi.org/project/muscriptor/
- muscriptor GitHub: https://github.com/muscriptor/muscriptor
- muscriptor HuggingFace: https://huggingface.co/MuScriptor
- muscriptor 论文: arXiv:2607.08168（Rouard et al., 2026）
- mujik-transcriptor 集成代码: `src/mujik/transcribe/muscriptor_adapter.py`
- 配置: `src/mujik/config/schema.py:TranscribeConfig.mode`
- 管线: `src/mujik/pipeline.py` (multitrack 分支)
- CLI: `src/mujik/cli.py:cmd_multitrack`

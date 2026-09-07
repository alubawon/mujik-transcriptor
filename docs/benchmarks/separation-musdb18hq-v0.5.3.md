# 分离质量基线：MUSDB18-HQ test 全量 50 首（v0.5.3，待审阅）

> **状态：候选基线，待用户审阅。** 审阅通过后作为 v0.5.3 版本的分离质量基线，
> 后续换模型/调参与此对照，退化即报警。
>
> 评测日期：2026-09-06/07 · 机器：Apple Silicon (MPS) · 协议：museval 标准窗口
>
> **本次评测是人工触发的（bench 不自动跑）**：仅人工触发或明确要求自动跑时执行。

## TL;DR

**`htdemucs_ft`（4-stem）全量胜出，维持主线默认；`htdemucs_6s`（6-stem）不换。**

| Stem | htdemucs_ft SDR | htdemucs_6s SDR | Δ | ft 胜率 |
|---|---|---|---|---|
| vocals | **9.62** | 8.96 | +0.66 | 35/50 |
| drums | **10.00** | 9.24 | +0.76 | 48/50 |
| **bass** | **8.83** | 8.06 | +0.77 | **47/50** |
| other | **6.80** | 6.05 | +0.75 | 46/50 |
| **均值** | **8.81** | 8.08 | +0.73 | — |

bass/other 混淆疑虑的检验结果：**在 MUSDB18-HQ（流行/摇滚为主）上，
4-stem 的 bass 反而比 6-stem 更准**——6-stem 多拆 piano/guitar 并没有换来
bass/other 的精度提升。之前 3 首抽样得出的 ft 优势（bass +1.27）在全量下收敛为
+0.77，方向一致。

## 1. 指标含义（面向阅读者）

### SDR（Signal-to-Distortion Ratio，信号失真比）——主指标

分离出的音轨与 ground truth 的整体相似度（dB），对数尺度：**+3dB ≈ 失真能量减半**。
包含全部误差来源（串音 + 干扰 + 算法噪声）。经验参考：

| SDR | 主观感受 |
|---|---|
| < 0 dB | 分离产物比直接用原曲还差（串音比目标还响） |
| 0–3 dB | 能听出目标，但串音严重 |
| 3–7 dB | 可用但不干净，适合粗转录 |
| 7–10 dB | 干净，适合下游转录（我们的工作区间） |
| > 10 dB | 非常干净（人声/鼓的顶尖水平） |

musdb 官方 leaderboard 上 SOTA 模型（含 htdemucs_ft 本身）的典型量级就是
vocals ~9-10 / drums ~10 / bass ~8.5-9 / other ~6-7 dB——本次结果与官方量级
一致，说明评测协议无误。**other 永远最低**：它包罗键盘/吉他/弦乐等一切其余乐器，
是最难的 stem。

### SIR（Signal-to-Interference Ratio，信号干扰比）

串音程度：其他 stem 的声音漏进本 stem 有多少。SIR 高 = 干净地"只含目标乐器"。
对转录最关键——**bass stem 里混进吉他的低音会直接变成错音符**。

### SAR（Signal-to-Artifacts Ratio，信号伪影比）

算法自身引入的噪声/伪影（不是串音）。SAR 低 = 分离过程本身产生了杂音。

### 聚合方式：per-track median → 跨 track median（musdb 官方口径）

museval 对每首曲子按 1s 窗口算 SDR，先取**该曲窗口中位数**（抗局部爆点），
再对 50 首取**中位数**（抗个别离谱曲拖偏均值）。故表中"Mean SDR"实为
"4 个 stem 各自的中位数再平均"。

## 2. 评测协议

- **数据**：MUSDB18-HQ test 50 首（`~/datasets/musdb18-hq`，Zenodo record 3338373，
  WAV 无损版；仓库不携带数据）
- **评测器**：museval（sigsep 官方 BSSEval v4 实现，默认 1s 窗口）
- **口径对齐**：htdemucs_6s 输出 6 stem（vocals/drums/bass/piano/guitar/other），
  其 other 定义为"piano/guitar 之外的其余"；MUSDB GT 的 other **包含** piano+guitar。
  公平对比把 6s 的 piano+guitar 加回 other 再评（MUSDB 无 piano/guitar 单独 GT）。
  不对齐会出现 SDR≈0 / SAR 深负的假象（SIR 反而很高是口径错位的指纹）。
- **设备/精度**：MPS；htdemucs_ft 走 in-process fp16 路径，htdemucs_6s 走
  demucs CLI 子进程 fp32 路径（demucs v4 CLI 无 fp16 flag）。SDR 对精度不敏感，
  两条路径可比。
- **抽样规则**：全量 50 首，无抽样。（3 首随机抽样仅作打通测试，不作结论。）

## 3. 结果明细

### 3.1 SDR 分布（min / 中位 / max）

| Stem | htdemucs_ft | htdemucs_6s |
|---|---|---|
| vocals | -0.51 / 9.62 / 14.00 | 2.39 / 8.96 / 14.38 |
| drums | -3.11 / 10.00 / 15.92 | 4.37 / 9.24 / 15.51 |
| bass | -1.01 / 8.83 / 22.65 | -1.32 / 8.06 / 21.84 |
| other | 0.61 / 6.80 / 11.21 | -1.96 / 6.05 / 9.16 |

两模型都有极个别曲子翻车（bass 负分 = 该曲 bass 分离完全失败），属正常分布尾部。

### 3.2 ft 翻车最重的 5 首（bass 视角）

| Track | bass SDR |
|---|---|
| AM Contra - Heart Peripheral | -1.01 |
| Hollow Ground - Ill Fate | 2.78 |
| The Doppler Shift - Atrophy | 2.90 |
| Timboz - Pony | 2.99 |
| Juliet's Rescue - Heartbeats | 3.03 |

### 3.3 per-track 全量数据

见 `sep_bench_full_htdemucs_ft.json` / `sep_bench_full_htdemucs_6s.json`
（含每首的 duration / sep_time / eval_time / 三指标分数）。

## 4. 耗时记录（后续分析的入口）

| 项 | htdemucs_ft | htdemucs_6s |
|---|---|---|
| 分离 mean (median) | 318s (312s) | 65s (62s) |
| museval 评估 mean | 375s | 424s |
| **50 首总墙钟** | **9.65h**（sep 4.4h + eval 5.2h） | **6.83h**（sep 0.9h + eval 5.9h） |
| 分离实时率 | 4.05x | 4.05x |

关键发现：

1. **museval 评估是绝对瓶颈**（占总墙钟 53-87%）。今后批量 bench 的优化入口：
   评估并行化（museval 单进程逐 track 串行）或窗口降采样（会牺牲与历史结果的可比性，需记录）。
2. ft 的 fp16-in-process 与 6s 的 fp32-CLI 实时率相同（4.05x）——之前 3 首抽样时
   "CLI fp32 比 in-process fp16 快 7 倍" 的印象来自 6s 模型本身更小，**不是精度差异**。
3. 墙钟记录均为本机 MPS；CUDA 容器上的耗时不可直接对比。

## 5. 结论与边界

**结论（候选基线）**：

1. 主线分离模型维持 `htdemucs_ft`（4-stem），Roformer 无引入依据（MUSDB 域内
   已是 SOTA 量级，且未实现 adapter）
2. `htdemucs_6s` 保留为 opt-in（piano/guitar 分离对结构分析有额外价值，但整体
   SDR 全面更低）
3. 本数字作为 v0.5.3 分离质量基线；后续任何分离侧改动对照此表回归

**结论边界（重要）**：

- MUSDB18 以流行/摇滚/电子为主，**无 metal / jazz fusion / latin dance**。
  我们的三首目标曲——buhee（jazz fusion）、monn（epic metal）、danso（latin
  dance）——都在覆盖域之外，本基线对它们的迁移性**未验证**。
- bass/other 混淆在 metal（吉他与贝斯频段重叠）上是否比 MUSDB 更严重，
  需要真曲 GT 或人工听感验证（MUSDB 结论不能直接外推）。
- 分离质量 ≠ 转录质量：SDR +0.7 dB 对最终 MIDI 音符的意义要靠
  `mujik.benchmarks.runner`（note F1）进一步确认。

## 6. 复现

```bash
# htdemucs_ft（in-process fp16, MPS）
python -m mujik.benchmarks.separation --musdb-root ~/datasets/musdb18-hq \
    --is-wav --variant htdemucs_ft --device mps \
    --work-dir ~/datasets/sep_bench_full_ft \
    -o sep_bench_full_htdemucs_ft.md --json sep_bench_full_htdemucs_ft.json

# htdemucs_6s（CLI subprocess fp32, MPS；HF 需镜像时加 HF_ENDPOINT）
HF_ENDPOINT=https://hf-mirror.com python -m mujik.benchmarks.separation \
    --musdb-root ~/datasets/musdb18-hq --is-wav --variant htdemucs_6s --device mps \
    --work-dir ~/datasets/sep_bench_full_6s \
    -o sep_bench_full_htdemucs_6s.md --json sep_bench_full_htdemucs_6s.json
```

依赖：`pip install 'mujik-transcriptor[separation-bench]'`（musdb + museval）+ ffmpeg。
评测不自动跑：仅人工触发或明确要求自动跑时执行。

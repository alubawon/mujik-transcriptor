# 分离质量基线：MUSDB18-HQ test 全量 50 首（v0.5.3，待审阅）

> **状态：候选基线，待用户审阅。** 审阅通过后作为 v0.5.3 版本的分离质量基线，
> 后续换模型/调参与此对照，退化即报警。
>
> 评测日期：2026-09-06/07 · 机器：Apple Silicon (MPS) · 协议：museval 标准窗口
>
> **本次评测是人工触发的（bench 不自动跑）**：仅人工触发或明确要求自动跑时执行。
>
> **v2 修订（2026-09-01）**：撤销 v1 的"ft 全量胜出"结论。v1 用同名 `other`
> 对比两个 `other` 定义不同的模型，且把 6s 唯一的差异化产出（piano/guitar）
> 排除在计分之外——该对比对 6s 不公平。修订后结论限定在 vocals/drums/bass
> 三轨；piano/guitar 列为未测。见 §2.1、§5.1。

## TL;DR

**在 vocals / drums / bass 三轨上，`htdemucs_ft` 稳定优于 `htdemucs_6s`（+0.66~0.77 dB）。
`other` 一列不可跨模型解读，piano/guitar 分离质量本协议测不了——因此本报告
不能得出"ft 整体优于 6s"的结论。**

| Stem | htdemucs_ft SDR | htdemucs_6s SDR | Δ | ft 胜率 | 可比性 |
|---|---|---|---|---|---|
| vocals | **9.62** | 8.96 | +0.66 | 35/50 | ✅ 同定义 |
| drums | **10.00** | 9.24 | +0.76 | 48/50 | ✅ 同定义 |
| **bass** | **8.83** | 8.06 | +0.77 | **47/50** | ✅ 同定义 |
| other | 6.80 | 6.05 | +0.75 | 46/50 | ⚠️ **见 §2.1，不可直接解读** |
| piano | — | 产出 | — | — | ❌ **MUSDB 无 GT，未测** |
| guitar | — | 产出 | — | — | ❌ **MUSDB 无 GT，未测** |
| 三轨均值（可比部分） | **9.48** | 8.75 | +0.73 | — | ✅ |

bass/other 混淆疑虑的检验结果（**仅限 bass 轨**）：在 MUSDB18-HQ（流行/摇滚为主）上，
4-stem 的 bass 比 6-stem 高 +0.77 dB，47/50 胜——**多拆 piano/guitar 没有换来
bass 精度提升**。之前 3 首抽样得出的 ft 优势（bass +1.27）在全量下收敛为 +0.77，
方向一致。这一条结论是成立的。

**不成立的结论（v1 报告的错误，已撤销）**：不能用本表判定 ft 整体优于 6s，
理由见 §2.1。6s 的真实卖点是 piano/guitar 两轨，而本协议对它们**零测量**。

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
是最难的 stem。（注意：跨模型比 `other` 有陷阱，见 §2.1。）

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

### 2.1 为什么 `other` 一列不可跨模型解读（本报告最重要的限制）

两个模型的 `other` **不是同一个东西**：

| | `other` 的语义 | 含 piano/guitar 能量？ |
|---|---|---|
| MUSDB GT | 除 vocals/drums/bass 外的一切（键盘、吉他、弦乐……） | ✅ 包含 |
| htdemucs_ft | 同上，模型直接拟合这个定义 | ✅ 包含 |
| htdemucs_6s | piano/guitar **之外**的其余（真正的残余轨） | ❌ 已被拆走 |

直接同名对比会让 6s 的 piano/guitar 全部记成 interference（表现为 SDR≈0 / SAR 深负 /
SIR 反而很高——这组指纹就是口径错位）。所以本次评测把 **6s 的 piano+guitar 加回
other** 再评。但这个补救只是让分数不再崩，**并不使结论公平**：

1. 加回后测的是"6s 的 piano+guitar+残余 之和"有多接近 GT other，即
   **6s 被迫带着自己的中间产物回来跟 ft 的直接输出比**。ft 一步拟合的目标，
   6s 要靠三路相加复原——误差只会累积，结构上就吃亏。
2. 6s 的真实交付物是**分开的 piano 和 guitar 两轨**。MUSDB 对这两者
   **没有任何 GT**，所以 6s 唯一的差异化价值在本协议里**得分恒为 0**——
   不是"测出来不好"，是"根本没进入计分"。

**结论**：`other` Δ=+0.75 只能读作"ft 拟合 GT other 这个定义更准"，
**不能**读作"ft 对 piano/guitar 的处理比 6s 好"。后者需要 §5.1 的另一套评测。

### 2.2 配置与数据

- **数据**：MUSDB18-HQ test 50 首（`~/datasets/musdb18-hq`，Zenodo record 3338373,
  WAV 无损版；仓库不携带数据）
- **评测器**：museval（sigsep 官方 BSSEval v4 实现，默认 1s 窗口）
- **口径对齐**：见 §2.1（6s 的 piano+guitar 加回 other）
- **设备/精度**：MPS。**两条路径实际都是 fp32**——htdemucs_6s adapter 会把
  `cfg.precision` 的 fp16/bf16 降级为 fp32 并 warning（demucs v4 CLI 无该 flag），
  见 `src/mujik/separate/htdemucs_6s_adapter.py`。故本次对比**不存在精度差异**。
  存在的真实配置不对称是 `--segment`：6s 因 Transformer 训练段长 7.8s 上限被
  floor 到 **7s**，ft 用默认段长。
- **抽样规则**：全量 50 首，无抽样。（3 首随机抽样仅作打通测试，不作结论。）

## 3. 结果明细

### 3.1 SDR 分布（min / 中位 / max）

| Stem | htdemucs_ft | htdemucs_6s |
|---|---|---|
| vocals | -0.51 / 9.62 / 14.00 | 2.39 / 8.96 / 14.38 |
| drums | -3.11 / 10.00 / 15.92 | 4.37 / 9.24 / 15.51 |
| bass | -1.01 / 8.83 / 22.65 | -1.32 / 8.06 / 21.84 |
| other ⚠️ | 0.61 / 6.80 / 11.21 | -1.96 / 6.05 / 9.16 |

⚠️ other 行不可跨列解读（§2.1）。

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
2. **两条路径实际都是 fp32，实时率相同（4.05x）**。之前 3 首抽样时"CLI 比
   in-process 快 7 倍"的印象来自 6s 模型本身更小，**不是精度差异**；且在 MPS 上
   in-process fp16 实测反而更慢（kernel fallback）。
3. 墙钟记录均为本机 MPS；CUDA 容器上的耗时不可直接对比。

## 5. 结论与边界

**成立的结论（候选基线）**：

1. **vocals / drums / bass 三轨**：`htdemucs_ft` 稳定优于 `htdemucs_6s`
   （+0.66 / +0.76 / +0.77 dB，胜率 35 / 48 / 47 of 50）。主线默认维持
   `htdemucs_ft` 有依据。
2. 多拆 piano/guitar **没有**换来 bass 精度提升——bass/other 混淆的原始疑虑，
   在 MUSDB 域内不能靠换 6s 解决。
3. 本三轨数字作为 v0.5.3 分离质量基线；后续分离侧改动对照此表回归。

**未决的问题（本协议无法回答）**：

1. **piano / guitar 分离质量完全未测**（MUSDB 无这两轨 GT）。因此
   **不能**说 ft 优于 6s：6s 的差异化产出在本协议里没有计分项。
   主观听感（`~/datasets/sep_compare`，buhee/moon/dança 三首）的反馈是
   两模型的 piano/guitar 都**很差**——这是当前的主要短板，需要独立评测。
2. 精度不是本次差异的来源（§2.2：两边都 fp32）。真实的配置不对称是
   6s 的 `--segment` 被 floor 到 7s，影响量级未测。

## 5.1 piano/guitar 评测方案（待执行）

MUSDB 结构上无法回答这个问题，必须换数据集。已调研结论：

| 数据集 | piano/guitar GT | 协议 | 适用性 |
|---|---|---|---|
| MoisesDB | ✅ 真实录音，逐 stem | 非商业（本地评测可用，不随仓库分发） | **首选** |
| Slakh2100 | ✅ 合成，可自动评 | CC BY 4.0 | 备选（合成音色，迁移性弱） |
| MedleyDB | ✅ 少量 | CC BY-NC-SA | 规模不足 |

注意评测轴要改：ft **根本不产出** piano/guitar 两轨，所以问题不是
"谁的 piano 更好"，而是"**6s 的 piano/guitar 是否比没有更有用**"——
下游是转录，判据应是 note F1 而非只看 SDR。

## 5.2 更高质量分离模型调研结论（2026-09）

调研了 Roformer 家族（详见下）。**结论：代码许可全部 OK（MIT），但没有任何一个
Roformer 权重能通过我们的权重许可门槛。**

| 方案 | 代码协议 | 权重协议 | stems | MUSDB SDR | MPS |
|---|---|---|---|---|---|
| lucidrains/BS-RoFormer | MIT ✅ | 不含权重 | — | — | — |
| MSST BS Roformer 4-stem | MIT ✅ | **未声明** 🚩 | v/d/b/o | 9.65 | 未验证 |
| MSST **SCNet XL IHF** | MIT ✅ | **未声明** 🚩 | v/d/b/o | **10.08** | 未验证 |
| audio-separator | MIT ✅ | **未声明** 🚩 | Roformer 仅 2-stem | — | 优秀 ✅ |
| bs-roformer-infer (SW) | MIT ✅ | **unknown + 来源链断裂** 🚩🚩 | **含 guitar/piano** | 仅 Multisong | 优秀 ✅ |

要点：

1. **BS-RoFormer 无一方权重**。ByteDance 只发论文（arXiv:2309.02612），
   代码是 lucidrains 复现（不含权重），权重全是社区训练的第三方 checkpoint——
   这是许可混乱的根因。
2. 唯一产出 guitar/piano 的 Roformer 是 **BS Roformer SW**（jarredou），
   MVSep Multisong 上 guitar 9.05 / piano 7.83，对比 htdemucs_6s piano **2.23**——
   潜在提升 ~5.6 dB，很诱人。但**原作者 HF 账号已注销（2026-06）**，
   现存镜像 `enerjazzer/BS-ROFO-SW-Fixed` 页面显示 "License: unknown"。
   sha256 能证明字节一致，**无法证明任何人有分发权**。
3. ZFTurbo/MSST 的 checkpoint 许可有三个 issue 追问（#245/#248/#249）
   至今**零回复**；且其 4-stem 仅用 MUSDB18-HQ train 训练，而 MUSDB 本身是
   非商业授权——即使作者回复，**他可能也没有立场给出宽松许可**。
4. ⚠️ **我们现用的 htdemucs 权重可能同样不合规**：facebookresearch/demucs#327
   指出 Meta 预训练权重"仅供科研用途"，尽管代码是 MIT。**这不是 Roformer 独有
   的问题，是我们已经背着的风险。**
5. 若只求 4-stem SDR 而不执着 Roformer，**SCNet XL IHF（10.08）已超过
   BS Roformer（9.65）**，同样 MIT 代码、同样权重未声明。
6. ⚠️ 跨表可比性：MVSep/社区数字多在 MVSep 私有 Multisong 上测，
   与我们的 museval MUSDB18-HQ 全曲口径**不可直接相减**，只能看方向。

**待决策（需用户定调）**：如果主线要商业分发，上述权重（含现用 htdemucs）
都不安全，出路是把分离做成用户自备权重的可选插件；如果研究/自用可接受，
`bs-roformer-infer --device mps` 是 guitar/piano 需求的技术最优解
（MIT 代码 + 真 CLI + 已验证 Apple Silicon + MLX 后端），按 madmom/BTC 的
先例做 subprocess 隔离，并**锁 HF commit revision 而非 main**。

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

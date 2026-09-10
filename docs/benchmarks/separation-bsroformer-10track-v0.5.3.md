# BS-Roformer SW 抽样评测（MUSDB18-HQ test 10 首，v0.5.3，待审阅）

> **状态：候选结论，待用户审阅。** 10 首抽样**不能**直接作版本基线
> （50 首全量才能，见 [demo 三曲曲风] 的抽样规则），但结论方向足够强，
> 可据此决定是否投入全量跑分。
>
> 评测日期：2026-09-01 · 机器：Apple Silicon (MPS) · 协议：museval 标准窗口
> · 抽样：seed=42, n=10 · **人工触发**

## TL;DR

**BS-Roformer SW 在同 10 首上对 `htdemucs_ft` 全面大幅领先：mean SDR
+2.80 dB，四轨逐曲胜率 10/10、10/10、9/10、9/10。这不是边际改进。**

| Stem | BS-Roformer | htdemucs_ft | htdemucs_6s | Δ (bsr−ft) | bsr 逐曲胜率 |
|---|---|---|---|---|---|
| vocals | **11.006** | 7.139 | 7.083 | **+3.87** | 10/10 |
| drums | **12.998** | 10.515 | 9.849 | **+2.48** | 10/10 |
| **bass** | **12.960** | 10.933 | 10.372 | **+2.03** | 9/10 |
| other ⚠️ | **9.597** | 6.761 | 6.096 | +2.84 | 9/10 |
| **均值** | **11.640** | 8.837 | 8.350 | **+2.80** | — |

⚠️ 三个模型的 `other` 语义不同（BS-Roformer SW 和 6s 都是 6-stem，
其 other 已剥离 guitar/piano，评测时加回；ft 的 other 原生包含）。
该列的可比性限制与主基线报告 §2.1 完全相同，**不可单独解读**。
但 vocals/drums/bass 三轨是同定义的，**+3.87/+2.48/+2.03 是干净的对比**。

### 为什么必须同曲目对比

本次是 10 首抽样，而 ft/6s 的既有结果是 50 首全量。**跨样本比中位数是错的**：

| | 同这 10 首 | 全量 50 首 |
|---|---|---|
| ft mean SDR | 8.837 | 8.811 |
| 6s mean SDR | 8.350 | 8.077 |

数字接近纯属巧合——**分轨看差异很大**（这 10 首的 ft vocals 中位数只有
7.139，而全量是 9.622，说明抽到的曲目人声明显更难）。所以表中 ft/6s 一列
是**从全量 50 首结果里挑出这 10 首重新算中位数**得到的，不是全量数字。

## 1. 这解决了什么已知问题

用户主观听感反馈的三个症状，对照本次数字：

| 症状 | 本次结果 |
|---|---|
| 鼓/人声"都可接受" | vocals +3.87 / drums +2.48，进一步大幅改善 |
| **bass 分离稍差** | **bass +2.03 dB，9/10 胜**——直接命中该症状 |
| **guitar/piano 还是很差** | ⚠️ **本协议仍无法回答**（MUSDB 无 GT），见 §3 |

bass 这一条尤其值得注意：主基线报告的结论是"多拆 piano/guitar 换不来
bass 精度提升"（6s 反而比 ft 差 0.77）。BS-Roformer SW **同样是 6-stem，
却在 bass 上比 4-stem 的 ft 高 2.03 dB**——说明此前的 bass 短板是
**模型能力问题，不是 6-stem 拆法本身的问题**。

## 2. 耗时

| 项 | BS-Roformer SW | htdemucs_ft（全量口径） |
|---|---|---|
| 分离 mean | 355s | 318s |
| 分离实时率 | **1.15x**（三曲实测） | 4.05x |
| 10 首总墙钟 | 6502s（sep 3554 + eval 2926） | — |

**BS-Roformer 反而更快**（实时率 1.15x vs 4.05x，即约 3.5 倍速度优势），
同时多产出 guitar/piano 两轨。质量和速度**没有 trade-off**。
（另有 MLX 后端可再快 ~2.5x 且省一半内存，未启用。）

## 3. 仍未解决：guitar/piano 无法定量

BS-Roformer SW 产出独立 guitar/piano，但 **MUSDB 对这两轨没有任何 GT**，
所以本次评测对它们**零测量**——与主基线报告 §5.1 是同一个 gap。

外部参考（**不可与本表相减**，MVSep 私有 Multisong 口径）：
BS-Roformer SW guitar 9.05 / piano 7.83，而 htdemucs_6s piano 仅 **2.23**
——若该量级可信，piano 提升约 5.6 dB。但这需要我们自己验证。

**下一步**：三首目标曲的三方听感对照已就绪（§5），定量则需 MoisesDB。

## 4. 权重许可（决策阻塞点）

⚠️ **代码 MIT，权重许可未声明。** 原作者 jarredou 的 HF 账号已于 2026-06
注销，现存镜像 `enerjazzer/BS-ROFO-SW-Fixed` 页面显示 "License: unknown"。
sha256 能证明字节一致，**无法证明任何人有分发权**。

adapter 已按 fail-loud 原则处理：每次调用打 license warning，
`Stem.metadata` 记录 `weights_license="unknown"`。

注：**我们现用的 htdemucs 权重亦非完全干净**（facebookresearch/demucs#327：
Meta 预训练权重"仅供科研用途"，尽管代码 MIT）。这不是引入 BS-Roformer
才产生的新风险。详见主基线报告 §5.2。

**因此本报告只推荐到"本地评测/研究用"这一步**；是否进主线默认路径
取决于分发场景，需用户定调。

## 5. 三曲听感对照（已生成，待主观评价）

`~/datasets/sep_compare/<track>/bsroformer/` — buhee（jazz fusion）、
moon（epic metal）、dança（latin dance），每首 6 stem + instrumental。
与既有 `ft/`（4 stem）、`6s/`（6 stem）并列，可三方 A/B。

这三首曲风都在 MUSDB 覆盖域外，**是本次数字迁移性的关键检验**。

## 6. 复现

```bash
# 独立 venv（依赖与主线冲突，故隔离）
uv venv bsrofo-venv
uv pip install --python ./bsrofo-venv/bin/python bs-roformer-infer

# 10 首抽样评测
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
MUJIK_BSROFORMER_BIN=$PWD/bsrofo-venv/bin/bs-roformer-infer \
HF_ENDPOINT=https://hf-mirror.com \
.venv/bin/python -m mujik.benchmarks.separation \
    --musdb-root ~/datasets/musdb18-hq --is-wav \
    --variant bsroformer --device mps --sample 10 --seed 42 \
    --work-dir ~/datasets/sep_bench_bsr10 \
    -o docs/benchmarks/sep_bench_bsr10.md \
    --json docs/benchmarks/sep_bench_bsr10.json
```

per-track 全量数据见 `sep_bench_bsr10.json`。
评测不自动跑：仅人工触发或明确要求自动跑时执行。

## 7. 建议

1. **听感对照优先**（§5 已就绪）——三首目标曲都在 MUSDB 域外，
   若听感与数字一致，结论就稳了。
2. 若听感确认，**跑 50 首全量**再作版本基线（10 首不够）。
3. 权重许可问题需用户决策（§4），与技术结论解耦。

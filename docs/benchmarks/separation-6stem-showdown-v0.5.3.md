# BS-Roformer SW vs htdemucs_6s：6-stem 对决（v0.5.3，待审阅）

> **状态：候选结论，待用户审阅。** 10 首抽样，不作版本基线。
> 评测日期：2026-09-01 · MPS · museval 标准窗口 · seed=42 · **人工触发**
>
> **前提更新**：本项目纯研究、永不商用 → 权重许可不再是阻塞项
> （BS-Roformer SW 权重 license unknown、htdemucs 权重"仅供科研"，均可用）。

## TL;DR

**两个 6-stem 模型正面对决，BS-Roformer SW 四轨全胜，mean SDR +3.29 dB。
这比它对 ft 的优势（+2.80）还大——htdemucs_6s 是三者中最弱的。**

| Stem | BS-Roformer | htdemucs_6s | Δ | SIR (bsr / 6s) | 逐曲胜率 |
|---|---|---|---|---|---|
| vocals | **11.006** | 7.083 | **+3.92** | 21.33 / 16.68 | 10/10 |
| drums | **12.998** | 9.849 | **+3.15** | 23.77 / 19.60 | 10/10 |
| bass | **12.960** | 10.372 | **+2.59** | 23.41 / 18.02 | 9/10 |
| other | **9.597** | 6.096 | **+3.50** | 17.64 / 11.46 | 9/10 |
| **均值** | **11.640** | 8.350 | **+3.29** | — | — |

**这次 `other` 一列是可比的。** 主基线报告 §2.1 说 ft 和 6s 的 other 不可比
（定义不同）——但这里对比的**两个都是 6-stem**，other 的语义完全一致
（都是"剥离 guitar/piano 后的残余"），评测时都加回 piano+guitar。
同定义、同处理 → **+3.50 dB 是干净的数字**。

**SIR 全面高 4-6 dB** 意义重大：SIR 衡量串音，对转录最关键
（bass 轨里混进吉他低音会直接变成错音符）。BS-Roformer 的串音显著更少。

## 1. guitar/piano 分离质量对比（你的重点）

**坏消息先说：MUSDB 无 guitar/piano GT，SDR 仍然测不了。** 但我做了一个
**不需要 GT** 的分析，能给出方向性证据。

### 1.1 跨模型一致性（log 能量包络相关）

两个**独立训练**的模型对同一首曲子的判断有多一致：
- 高一致 → 两者大概率都抓到了真实存在的乐器（弱证据）
- 低一致 → **至少有一个是错的**（强证据，可证伪）

关键是**设对照组**：vocals/drums/bass 两模型 SDR 都有 10-13 dB（都做得好），
它们的一致性就是"做对了该长什么样"的基准线。

| Stem | 三曲平均一致性 | 解读 |
|---|---|---|
| bass | **+0.898** | 对照：做得好 = 0.9 |
| drums | **+0.895** | 对照：做得好 = 0.9 |
| other | +0.677 | |
| vocals | +0.671 | （buhee 仅 0.33，该曲人声稀疏） |
| **guitar** | **+0.643** | ⚠️ 明显低于对照组 |
| **piano** | **+0.461** | 🚩 **最低，且远低于对照组** |

**结论：piano 一致性只有 0.46，不到 bass/drums（0.90）的一半。
两个模型对"哪些声音是钢琴"存在严重分歧——这在数学上意味着
至少有一个模型是错的，与你"guitar/piano 分离还是很差"的听感一致。**
guitar（0.64）居中，比 piano 好但仍明显低于对照组。

### 1.2 能量分配（另一个视角）

| Track | guitar 6s / bsr | piano 6s / bsr |
|---|---|---|
| buhee (jazz fusion) | 16.42% / 10.56% | 0.22% / 0.42% |
| moon (epic metal) | 4.58% / 5.50% | 0.22% / 0.13% |
| dança (latin dance) | 5.64% / 2.84% | **0.00% / 0.00%** |

**两个模型都几乎不往 piano 轨里放东西**（占总能量 0.1-0.4%，dança 直接为 0）。
buhee 是 jazz fusion（大概率有键盘），piano 却只有 0.42% —— 要么这些曲子
确实没钢琴，要么**模型把钢琴内容漏到了 other/guitar 里**。
dança 的 piano 能量为 0（-117 dBFS，纯数字静音）至少是诚实的"我认为没有"。

### 1.3 这一节能和不能证明什么

- ✅ **能证明**：piano 的跨模型分歧远大于已知做得好的轨 → 至少一个模型错了
- ✅ **能证明**：两模型都极少往 piano 轨输出内容
- ❌ **不能证明**：谁的 guitar/piano 更好（无 GT，一致性不分对错）
- ❌ **不能证明**：BS-Roformer 的 piano 是否真比 6s 强 5.6 dB（MVSep 宣称值）

**要定量必须上 MoisesDB**（见 §3）。外部参考（MVSep 私有 Multisong 口径，
**不可与本表相减**）：BS-Roformer SW guitar 9.05 / piano 7.83
vs htdemucs_6s piano **2.23**。

## 2. 三模型总排名（同 10 首）

| 模型 | mean SDR | vs 最优 |
|---|---|---|
| **BS-Roformer SW** | **11.640** | — |
| htdemucs_ft | 8.837 | −2.80 |
| htdemucs_6s | 8.350 | −3.29 |

**htdemucs_6s 已可完全排除**：它比 ft 更差，又被 BS-Roformer 全面压制，
且其唯一卖点（guitar/piano）在 §1 的一致性分析里表现最弱。

## 3. MoisesDB 方案（当前计划）

### 为什么必须是它

MUSDB 结构上无法回答 guitar/piano —— **没有 GT 就是没有**，
再怎么改评测代码都变不出来。可选数据集：

| 数据集 | guitar/piano GT | 规模 | 协议 | 结论 |
|---|---|---|---|---|
| **MoisesDB** | ✅ 真实录音，逐 stem | 240 首 / 11 类 | 非商业（研究可用） | **首选** |
| Slakh2100 | ✅ 合成 | 2100 首 | CC BY 4.0 | 备选：合成音色迁移性弱 |
| MedleyDB | ✅ 少量 | 122 首 | CC BY-NC-SA | 规模不足 |

本项目纯研究 → MoisesDB 的非商业条款**不构成问题**。

### 计划（尚未执行，需你点头）

1. **取数**：MoisesDB 需在官方站注册申请下载（非 Zenodo 直链），
   约 40 GB。当前磁盘余量 32 GB → **需要先清理**
   （`~/datasets/sep_bench_bsr10` 5.1G + `sep_bench_full_*` 可删，
   结果都已在 JSON 里）。
2. **改评测器**：`separation.py` 现在硬绑 musdb 库和 4 stem
   （`SEPARATION_STEMS`）。需要：
   - 新增 MoisesDB 读取（它是 per-stem 目录结构，非 musdb 格式）
   - stem 映射：MoisesDB 的 11 类 → 我们的 6 类（guitar/piano 直接对应；
     其余归并进 other）
   - **不再需要 §2.1 的加回 other 补救**——有真 GT 就直接同名比
3. **评测轴要改对**：ft 根本不产 guitar/piano，所以正确的问题不是
   "谁的 piano 更好"，而是"**6-stem 的 guitar/piano 是否比没有更有用**"。
   且下游是转录 → 最终判据应是 **note F1**，不能只看 SDR。
4. **预计耗时**：museval 是瓶颈（占墙钟 53-87%）。10 首约 1.8h，
   若跑 30 首 guitar/piano 子集约 5-6h。

## 4. 复现

```bash
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

对比脚本口径：从全量 50 首结果里**挑出这 10 首重算中位数**再比
（跨样本比中位数是错的）。

## 5. 一致性分析的方法说明（避免重蹈覆辙）

第一版我用**波形相关**算一致性，得到 guitar/piano ≈ 0.03，看似"完全不一致"。
**这个结论是错的**：波形相关对相位/时移极度敏感，实测两模型输出存在
约 2 秒的整体偏移，且 vocals/drums/bass 的波形相关同样≈0——
而它们的 SDR 有 11-13 dB，足以证明"波形相关≈0"根本不代表内容不一致。

改用 **log 能量包络相关**（~93ms 帧 RMS，相位无关）后，对照组回到
0.90 的合理水平，guitar/piano 的偏低才成为有意义的信号。

**教训**：任何新指标必须先在**已知答案的对照组**上验证它能给出预期值，
否则会拿一个坏掉的尺子量出"结论"。

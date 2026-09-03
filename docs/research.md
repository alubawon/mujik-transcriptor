# mujik-transcriptor 研究报告

> 版本：v0.1（2026-08-26）
> 方法：deep-research 工作流（5 角度并发 + 25 源抓取 + 3 轮对抗验证 + 5 条幸存）

## 0. 摘要

本研究从 5 个角度对端到端音乐音频处理管线做选型评估。

经 3 轮对抗验证，**25 条候选主张中 5 条幸存**（confidence: high × 3 / medium × 2），其余 20 条被驳回（多数为"接近 SOTA"型未交叉验证数字）。

**主干结论**：
- ✅ **源分离**：Demucs v4 htdemucs/htdemucs_ft（3-0 验证，9.20 dB SDR on MUSDB18-HQ）
- ✅ **节拍/下拍**：Beat Transformer（offline）+ BeatNet（online）
- ⚠️ **时间签名**：ResNet18/METER2800 限 3/4/4/4/5/4/7/8
- 🧪 **复调转录 / 鼓转录 / 和弦 / 演奏细节保留 / PDF 排版**：需本地 benchmark

**v2 调整（基于 4 条用户澄清）**：
- 许可证：主线 MIT/Apache/BSD；GPL 进程隔离为微服务；避开 CC-BY-NC/AGPL
- 拍号：分段一等公民 + 改拍号两种堆积模式
- 分轨：4/5/6-stem 可插拔，主线 Demucs v4，扩展 Roformer 家族
- 开发环境：uv + Docker + GPU 容器 + 模型权重不入仓

## 1. 5 条幸存主张

### 1.1 Demucs v4 htdemucs 架构（high, 3-0）

**Claim**：Demucs v4 htdemucs 是一种混合时域/频域 bi-U-Net，最内层被跨域 Transformer Encoder 替换；训练数据为 MUSDB18-HQ 加 800 首人工策展曲目，44.1 kHz 立体声。

**证据**：Meta 论文《Hybrid Transformers for Music Source Separation》（Defossez et al., 2022, arXiv:2211.08553）+ 官方 facebookresearch/demucs 仓库 README。架构定义（保留最外 4 层 bi-U-Net，替换最内 2 层 encoder/decoder 为 depth=5/8-head/dim=384 的跨域 Transformer）与数据描述（MUSDB18-HQ + 从 3500 首经 P[i,i]>70%、P[i,j]<30% 自动过滤的 800 首内部曲目，44.1 kHz 立体声，8x V100 32GB，fp32，1200 epoch × 800 batch）逐项可核。原始 htdemucs 在 MUSDB18-HQ 上 SDR ≈9.0 dB，htdemucs_ft 提升到 9.20 dB。

**来源**：
- https://github.com/facebookresearch/demucs
- https://arxiv.org/abs/2211.08553

### 1.2 Beat Transformer Ballroom F-Measure（high, 3-0）

**Claim**：在 Ballroom 数据集上，Beat Transformer 的 Beat F-Measure 0.968 / Downbeat F-Measure 0.941，超过 TCN+Demix 基线（0.960/0.925）和 Böck 等（0.962/0.916），主要增益来自 downbeat（+4 pp）。

**证据**：数值直接出自 arXiv:2209.07140v1 表 1 Ballroom 行，与 HTML 版本完全一致。论文为 ISMIR 2022 同行评审（Zhao/Xia/Wang），且其文字明确指出"downbeat 改进比 beat 改进更显著"，与"4 pp downbeat 提升"一致。

**来源**：
- https://arxiv.org/abs/2209.07140
- https://arxiv.org/html/2209.07140v1

### 1.3 BeatNet 在线联合跟踪（high, 3-0）

**Claim**：BeatNet 是首个在线联合 beat/downbeat/meter 跟踪系统，采用因果 CRNN 加推理阶段 Sequential Monte Carlo 粒子滤波，无需预先指定拍号。

**证据**：四项子主张均有可核来源：
1. "Online joint beat/downbeat/meter"——官方仓库描述 + arXiv:2108.03576
2. "因果 CRNN"——SingNet 论文 arXiv:2306.02372 明确指出
3. "推理阶段 SM-C 粒子滤波"——SingNet 论文与官方 inference_model='PF' 参数双向印证
4. "无需拍号 priming"——ar5iv 镜像包含原文 "No time signature priming required"

**来源**：
- https://github.com/mjhydri/BeatNet
- https://ar5iv.labs.arxiv.org/html/2108.03576
- https://arxiv.org/abs/2306.02372

### 1.4 时间签名模型范围（medium, 2-1）

**Claim**：基于 ResNet18 + MFCC 的时间签名检测模型只覆盖 3/4、4/4、5/4、7/8 四种拍号，并在单数据集 Meter2800 上训练，对 11/8、13/8、变拍子与 progressive metal/math-rock/free jazz 中的 metric modulation 等未见验证。

**证据**：DOI 10.1186/s13636-024-00346-6（EURASIP J. Audio Speech Music Processing, 2024）正文明确说"关注四种主要节拍：3、4、5、7"；配套数据集论文 PMC10700346 确认 4 类设计且分布不均。独立文献 arXiv:2502.12972（2025）也证实公共时间签名模型在少见拍子上表现退化。

**来源**：
- https://link.springer.com/content/pdf/10.1186/s13636-024-00346-6.pdf
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10700346/
- https://arxiv.org/abs/2502.12972

### 1.5 BTC-HCQT 与 BTC 持平（medium, 2-1）

**Claim**：btc-hcqt 是 BTC 的 HCQT 前端变体，在公开 GuitarSet 与 Schubert 上与基线 BTC 持平：80.5/63.0 vs 80.9/64.6（GuitarSet root/7ths），73.8/55.6/65.3 vs 73.1/55.3/64.1（Schubert root/7ths/mirex），所有差距均在 95% 置信区间内。

**证据**：所有数字与 README/BENCHMARK.md 一致；方法学（95% CI via 模型标准误）在作者贴文中给出。仓库仍提交，MIT 许可。属于"持平"型负结果，避免 SOTA 过度宣称。

**来源**：
- https://github.com/marcusfkelley/btc-hcqt
- https://selektaudio.com（作者贴）

## 2. 逐环节对比

### 2.1 源分离

| 工具 | 输出 | 许可证 | 速度（GPU） | SDR | 推荐 |
|---|---|---|---|---|---|
| **Demucs v4 htdemucs_ft** | 4-stem | MIT | 单 GPU 实时+ | 9.20 dB | ⭐ 首选 |
| Demucs v4 htdemucs | 4-stem | MIT | 略快 | ~9.00 dB | 备选 |
| Demucs v4 mdx_q | 4-stem | MIT | CPU 实时 | 略低 | 备选（无 GPU） |
| Spleeter | 2/4/5-stem | MIT | GPU 快 | 6.x dB | 不推荐 |
| Open-Unmix | 4-stem | MIT | GPU 中 | 6.x dB | 备选（学术） |
| **MDX-Net (Kuielab)** | 多模型 | Apache 2.0 | 极快 | 9.x dB | ⭐ 备选 |
| **BS-Roformer (21X)** | 6+ stem | MIT | 中 | 9.x+ dB | ⭐ 扩展 5/6-stem |
| **Mel-Band-Roformer** | 6+ stem | MIT | 中 | 高 | ⭐ 扩展 5/6-stem |
| AudioSep | 文本引导 | MIT | 中 | 通用但非专用 | 备选 |

**关键事实**：
- Demucs v4 的"other" 轨是残差聚合（吉他、钢琴、合成器等都进这里），SDR 比 vocals/drums/bass 低 2-3 dB
- 5/6-stem 需 Roformer 家族或 MDX23C
- htdemucs_ft 训练依赖 Meta 内部 800 首人工策展曲，但权重可直接推理

### 2.2 转录（旋律/钢琴/吉他/贝斯多音）

| 工具 | 输出 | 许可证 | 复调 | 速度 | 推荐 |
|---|---|---|---|---|---|
| **basic-pitch (Spotify)** | note + onset/offset/velocity | Apache 2.0 | 中 | 极快 | ⭐ 首选（轻量） |
| **ByteDance piano_transcription** | onset/offset/velocity + 踏板 | Apache 2.0 | 高 | 中 | ⭐ 钢琴首选 |
| **Apollo (SONDREAM)** | 多轨多音 | MIT | 高 | 中 | ⭐ 综合备选 |
| MT3 (Magenta) | 多乐器多音 | Apache 2.0 | 高 | 重 | 备选（重） |
| Transkun V2 | 钢琴 onset/offset | MIT | 中 | CPU 实时 | 备选（轻量） |
| Onsets and Frames | 钢琴 | Apache 2.0 | 中 | 重 | 经典基线 |
| Omnizart | 一站式 | MIT | 中 | 中 | 备选（精度中等） |

**已知风险**：
- basic-pitch 的 12/19/24 谐波"幻觉"——会把低频泛音错识别成新的 note
- basic-pitch 不支持弯音（pitch bend），需 CREPE/pyworld 提 f0 后单独注入
- "保留演奏细节"目标与 basic-pitch + pretty-midi 默认模型不完全对齐

### 2.3 鼓转录

> **2026-09 复查更正**：本表当年标的 adtof "MIT" 有误——原仓库 MZehren/ADTOF 实为
> CC-BY-NC-SA 4.0（非商用），且 `Music-and-Culture-Technology-Lab/Adtof` URL 已 404，
> PyTorch 移植 xavriley/ADTOF-pytorch 无 LICENSE。v0.5.2 起主线改用 DrumScript（下表）。

| 工具 | 输出类别 | 许可证 | 训练集 | 推荐 |
|---|---|---|---|---|
| **DrumScript** | kick/snare/hh open+closed/3 tom/crash/ride | Apache-2.0 | 无（规则引擎，素材以技术死亡金属为主） | ⭐ v0.5.2 主线 |
| ~~adtof~~ | 5 类 | CC-BY-NC-SA（原表误标 MIT） | ADTOF (RockBand 谱面) | ❌ 弃用：死链 + NC + 移植版无 LICENSE |
| **Omnizart (drum module)** | 多类 | MIT | A2MD | 🧪 备选 |
| YODO (YOLOv4) | 11 类 | MIT | E-GMD | 🧪 备选（单人项目停更风险） |

**已知风险**：
- ADTOF 训练集来自 RockBand 谱面，对真实金属失真、双底鼓 blast beat、ghost note 鲁棒性未交叉验证
- DrumScript 为 alpha 规则引擎：速度优先于准确率，jazz/funk 鼓（ghost note 密集场景）官方自述准确率一般；kick 偏少 / hi-hat 偏多（buhee 实测 67 kick vs 1043 closed-hh）

### 2.4 节拍/下拍跟踪

| 工具 | 模式 | 许可证 | 速度 | 推荐 |
|---|---|---|---|---|
| **Beat Transformer** | offline | MIT | 中 | ⭐ 首选（高准确） |
| **BeatNet** | online + offline | MIT | 极快 | ⭐ 备选（在线） |
| madmom | offline | BSD-3 | 中 | 不推荐（已停滞） |
| librosa.beat.beat_track | offline | ISC | 极快 | 极简基线 |
| Essentia | offline | AGPL-3.0 ⚠️ | 中 | 注意 AGPL 传染 |

### 2.5 时间签名

| 工具 | 覆盖拍号 | 数据集 | 推荐 |
|---|---|---|---|
| **ResNet18 + MFCC (EURASIP 2024)** | 3/4 / 4/4 / 5/4 / 7/8 | METER2800 | ⭐ 唯一开箱即用 |
| Beat Transformer (time-signature head) | 隐式 | — | 备选（未独立验证） |

**已知风险**：
- 仅覆盖 4 类拍号，对 11/8、13/8、变拍子、metric modulation 全部失效
- 流行+爵士+金属中常见 5/4 progressive metal、7/8 摇滚、混合拍子 jazz 会大量误判

### 2.6 和弦识别

| 工具 | 输出 | 许可证 | 推荐 |
|---|---|---|---|
| **BTC-HCQT** | root + 7ths | MIT | ⭐ 基线 |
| **CREMA** | root + quality | MIT | ⭐ 备选 |
| **Chord-CNN-LSTM** | root + triad/7/9/11/13 + bass | MIT | ⭐ 备选（最细粒度） |
| Chordino (aubio) | root | GPL-3.0 ⚠️ | 避开（GPL 传染） |
| madmom (ChordRecognitionProcessor) | root + bass + 7ths | BSD-3 | 备选（停滞） |

**v0.5.2 落地方式（BTC-HCQT vendoring）**：BTC-ISMIR19（Park et al., ISMIR 2019，
MIT，[jayg996/BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19)）的推理代码
vendor 进 `src/mujik/chord/_btc/`（btc_model + utils 三件套 + LICENSE；
170 类词表代码内生成无数据文件；本地补丁：np.float→float，numpy 1.24+）。
large_voca 权重（12MB，随上游仓库分发）在 ml 镜像构建时下载到
`/app/models/`，运行时按 `config.btc_model_path` → env `MUJIK_BTC_MODEL` 解析。
buhee×jazz 实测：234 chords，maj7/m7/7 延伸和弦占 171/234（madmom 时代仅 major/minor）。

**已知风险**：
- 领域 plateau 在 77-82% root accuracy
- 爵士 9/11/13 扩展和弦识别率低

### 2.7 MIDI 工具

| 工具 | 许可证 | 强项 | 推荐 |
|---|---|---|---|
| **pretty-midi** | MIT | note/control/乐器/速度读写 | ⭐ 主读写 |
| **mido** | MIT | 实时/流式、轻量、事件级编辑 | ⭐ 主网格化 |
| **music21** | BSD-3 | 乐理分析、和弦/罗马数字、量化 | ⭐ 主分析+转换 |
| miditoolkit | MIT | 多轨合并、tokenization | 备选 |

### 2.8 乐谱排版与 PDF

| 工具 | 许可证 | 强项 | 弱项 | 推荐 |
|---|---|---|---|---|
| **LilyPond** | GPL-2.0+ | 排版精细、复调/力度/踏板 | 学习曲线陡 | ⭐ 精细（GPL 隔离） |
| **MuseScore 4 CLI** | GPL-2.0+ | GUI/CLI 双模 | 依赖 Qt 库 | ⭐ 快速（GPL 隔离） |
| **Verovio** | BSD-3 | 轻量、headless | 复调+力度排版粗糙 | ⭐ 主线轻量 |
| **OSMD** | BSD-3 | 浏览器渲染 | 不支持 PDF 导出 | ⭐ Web 预览 |
| **music21 + LilyPond** | BSD-3 + GPL | music21 stream → lilypond | 复杂记谱需手工修 | ⭐ Python 友好 |

## 3. v1 → v2 关键变更

| 维度 | v1 | v2 |
|---|---|---|
| 许可证策略 | 未明确 | 主线 MIT/Apache/BSD；GPL 进程隔离 |
| 拍号处理 | 单一全局拍号 + 手动覆盖 | 分段一等公民 + 改拍号两种模式 |
| 分轨方案 | 默认 4-stem | 4/5/6-stem 可插拔 |
| 开发环境 | 未涉及 | uv + Docker + GPU + 模型权重分离 |
| 基础栈 | 未明确 | 主线 PyTorch；basic-pitch 的 TF 走子进程 |

## 4. 推荐组合

### 4.1 首选（v0.1 / v0.2）

```
pyloudnorm（响度归一）
  → deEchoes（可选去混响）
  → Demucs v4 htdemucs_ft（4-stem 源分离）
  → [vocals, bass, other] → basic-pitch
  → [drums] → adtof
  → Beat Transformer（节拍+下拍）
  → ResNet18/METER2800（时间签名，限 4 类）
  → BTC-HCQT + Chord-CNN-LSTM（和弦，optional）
  → mido + pretty-midi + music21（量化/对齐/合并）
  → Verovio（PDF，v0.1 占位）
```

### 4.2 精细版（v0.3 起）

```
[首选] → LilyPond 隔离服务 → PDF
```

### 4.3 扩展版（v0.4 起）

```
5/6-stem 启用：Demucs → MDX23C / BS-Roformer / Mel-Band-Roformer
piano 轨：ByteDance piano_transcription
guitar 轨：Apollo
```

## 5. 关键参考链接

### 源分离
- facebookresearch/demucs：https://github.com/facebookresearch/demucs
- Hybrid Transformers for Music Source Separation：https://arxiv.org/abs/2211.08553
- Kuielab MDX-Net：https://github.com/kuielab/mdx-net
- BS-Roformer (by 21X)：https://github.com/21X/MVSEP-MDX23-Colab_v2
- AudioSep：https://github.com/Audio-AGI/AudioSep

### 转录
- Spotify basic-pitch：https://github.com/spotify/basic-pitch
- ByteDance piano_transcription：https://github.com/bytedance/piano_transcription
- SONDREAM Apollo：https://github.com/sony/apollo
- Magenta MT3：https://github.com/magenta/mt3
- Transkun：https://github.com/yunchengle/Transkun
- Omnizart：https://github.com/Music-and-Culture-Technology-Lab/omnizart
- ADTOF 数据集：https://arxiv.org/abs/2107.05535

### 节拍/下拍/时间签名
- Beat Transformer：https://arxiv.org/abs/2209.07140
- BeatNet：https://github.com/mjhydri/BeatNet
- EURASIP 时间签名 ResNet18：https://link.springer.com/content/pdf/10.1186/s13636-024-00346-6.pdf
- METER2800 数据集：https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10700346/

### 和弦识别
- btc-hcqt：https://github.com/marcusfkelley/btc-hcqt
- CREMA：https://github.com/crema-music/crema
- Chord-CNN-LSTM：https://github.com/ptnghia-j/ChordMiniApp

### MIDI 处理
- pretty-midi：https://github.com/craffel/pretty-midi
- mido：https://github.com/mido/mido
- music21：https://web.mit.edu/music21/doc/

### 排版与导出
- LilyPond：https://lilypond.org/
- MuseScore：https://musescore.org/
- Verovio：https://www.verovio.org/
- OpenSheetMusicDisplay：https://opensheetmusicdisplay.org/

### 端到端参考
- musdb（源分离 benchmark）：https://github.com/sigsep/sigsep-mus-db

## 6. 本地 benchmark 清单

落地前在自家曲库上跑一遍（每项 5-10 首代表曲）：

1. **Demucs 4-stem 评估**：museval 在 5 类（流行/摇滚/爵士/金属/电子）各 2 首上算 SDR/SIR/SAR
2. **多音转录对比**：basic-pitch vs ByteDance piano vs Apollo 在爵士钢琴/失真吉他片段的 onset+offset+velocity F1
3. **鼓转录对比**：adtof vs Omnizart vs DrumTransformer 在金属 blast beat + 爵士 ghost note 片段的 F1
4. **时间签名边界测试**：11/8、13/8、变拍子片段的 ResNet18 输出 + downbeat 距离推断对比
5. **和弦扩展性测试**：爵士 maj9/m11/13/alt 和弦片段的 BTC-HCQT vs Chord-CNN-LSTM root/quality 准确率
6. **演奏细节保留**：原始音频 f0 vs 转录 MIDI note + 注入 pitch bend 的频谱差异
7. **PDF 排版对比**：LilyPond vs MuseScore vs Verovio 在同一段复调+力度+踏板的视觉评分

优先级 1+2+3+6 覆盖 80% 关键风险。

## 7. 风险登记

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

## 8. 免责与限制

1. 报告基于 5 条经 3 轮对抗验证的"幸存"主张，其余 19 条均被驳回或证据不足
2. 多个"接近 SOTA"型数值（SDR、F1、Chord 准确率）来自单篇论文/单数据集，未在所有 5 类目标音乐（流行/爵士/金属/古典/电子）上交叉验证
3. 维护状态/许可证基于 2024-2026 年的公开仓库快照，开源项目 license 与维护活跃度可能随时变化
4. 复调和金属鼓转录、演奏细节保留（弯音/滑音/力度）这三条是初选方案的公认弱项，本报告只给出风险标注，未做新实验
5. 时间签名模型对 11/8、13/8、变拍子、metric modulation 的泛化未验证，是流行+爵士+金属场景下最大未解难题

---

> 对应设计见 [design.md](./design.md)

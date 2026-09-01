#!/usr/bin/env bash
# scripts/run_demo.sh — 一键 demo：音频 → MIDI + 乐谱
#
# 用法（命令即教程）：
#   ./scripts/run_demo.sh buhee/buhee.mp3            # 仓库自带 demo 音频
#   ./scripts/run_demo.sh path/to/your_song.wav      # 自己的音频
#   ./scripts/run_demo.sh buhee/buhee.mp3 30         # 只跑前 30 秒（需 ffmpeg）
#   ./scripts/run_demo.sh buhee/buhee.mp3 "" pop     # 显式指定 preset（默认配置即 pop）
#
# 多 preset 对比（opt-in，开发/评测用）：
#   MUJIK_DEMO_PRESETS="pop,jazz,metal" ./scripts/run_demo.sh buhee/buhee.mp3
#
# 产物布局（曲名 = 输入文件名去扩展名）：
#   demo_out/<曲名>/
#   ├── project.mid          最终产物
#   ├── score.musicxml       最终产物
#   ├── score.pdf            verovio 可用时
#   ├── project.json         元数据
#   └── ws/                  中间产物（stems/tracks/beats.json ...）
#   多 preset 对比时：demo_out/<曲名>/<preset>/ + 共享 demo_out/<曲名>/ws
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- 0. 参数 ---
DEFAULT_WAV="$REPO_ROOT/buhee/buhee.mp3"
INPUT_WAV="${1:-$DEFAULT_WAV}"
DURATION_ARG="${2:-}"
PRESET_ARG="${3:-}"
OUT_ROOT="$REPO_ROOT/demo_out"
mkdir -p "$OUT_ROOT"

# --- 1. 校验输入 ---
if [[ ! -f "$INPUT_WAV" ]]; then
  if [[ "$INPUT_WAV" == "$DEFAULT_WAV" ]]; then
    echo "[demo] ❌ 默认 demo 音频不存在: $DEFAULT_WAV"
    echo "[demo] 请把 mp3/wav 放到 $DEFAULT_WAV，或显式传入: $0 your_song.wav"
  else
    echo "[demo] ❌ input not found: $INPUT_WAV"
  fi
  exit 1
fi

# 曲名 = 输入文件名去扩展名
SONG="$(basename "$INPUT_WAV")"; SONG="${SONG%.*}"
SONG_DIR="$OUT_ROOT/$SONG"
mkdir -p "$SONG_DIR"

# --- 2. 探测时长（soundfile → ffprobe → 跳过）---
DUR="?"
if python3 -c "import soundfile, sys; sys.stdout.write(f'{soundfile.info(\"$INPUT_WAV\").duration:.1f}')" 2>/dev/null; then
  DUR=$(python3 -c "import soundfile; print(f'{soundfile.info(\"$INPUT_WAV\").duration:.1f}')" 2>/dev/null)
elif command -v ffprobe >/dev/null 2>&1; then
  DUR=$(ffprobe -v error -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "$INPUT_WAV" 2>/dev/null || echo "?")
fi
echo "[demo] input: $INPUT_WAV (${DUR}s)  →  $SONG_DIR/"

# --- 3. 可选：按 DURATION_ARG 裁剪（确定性命名，同曲同长度可复用）---
WORK_WAV="$INPUT_WAV"
if [[ -n "$DURATION_ARG" ]]; then
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[demo] ⚠ ffmpeg 未装，跳过裁剪（用原文件）"
  else
    WORK_WAV="$OUT_ROOT/_trimmed_${SONG}_${DURATION_ARG}s.wav"
    ffmpeg -y -i "$INPUT_WAV" -t "$DURATION_ARG" -ar 44100 -ac 2 \
      "$WORK_WAV" >/dev/null 2>&1 || {
      echo "[demo] ❌ ffmpeg 裁剪失败"; exit 1;
    }
    echo "[demo] trimmed → ${DURATION_ARG}s"
  fi
fi

# --- 4. 环境探测 ---
HAS_MUJIK=0
command -v mujik >/dev/null 2>&1 && HAS_MUJIK=1
HAS_FFMPEG=0; command -v ffmpeg >/dev/null 2>&1 && HAS_FFMPEG=1
HAS_VEROVIO=0; command -v verovio >/dev/null 2>&1 && HAS_VEROVIO=1
python3 -c "import demucs" 2>/dev/null && HAS_DEMUCS=1 || HAS_DEMUCS=0
python3 -c "import madmom" 2>/dev/null && HAS_MADMOM=1 || HAS_MADMOM=0
echo "[demo] mujik=$HAS_MUJIK  ffmpeg=$HAS_FFMPEG  demucs=$HAS_DEMUCS  madmom=$HAS_MADMOM  verovio=$HAS_VEROVIO"

# --- 5. 跑管线（默认单 run；$3 显式单 preset；MUJIK_DEMO_PRESETS 多 preset 对比）---
# $3 显式 preset 也算"单 run"：最终产物直接在 曲名/ 下，preset 记录进 project.json
COMPARISON_MODE=0
PRESETS=()
if [[ -n "${MUJIK_DEMO_PRESETS:-}" ]]; then
  COMPARISON_MODE=1
  IFS=',' read -r -a PRESETS <<< "$MUJIK_DEMO_PRESETS"
elif [[ -n "$PRESET_ARG" ]]; then
  PRESETS=("$PRESET_ARG")
else
  PRESETS=("")
fi

run_once() {
  local preset="$1"
  local out_dir
  if [[ $COMPARISON_MODE -eq 1 ]]; then
    out_dir="$SONG_DIR/$preset"   # 对比模式：按 preset 分目录，共享 ws
  else
    out_dir="$SONG_DIR"           # 单 run：最终产物直接在 曲名/ 下
  fi
  local preset_args=()
  [[ -n "$preset" ]] && preset_args=(--preset "$preset")
  # 对比模式下共享同一 ws（中间产物只留一份）
  local ws_args=()
  if [[ $COMPARISON_MODE -eq 1 ]]; then
    ws_args=(--workspace "$SONG_DIR/ws")
  fi

  echo
  echo "── run → ${out_dir#$REPO_ROOT/} ${preset:+[preset=$preset]} ──"
  mkdir -p "$out_dir"
  # macOS bash 3.2 兼容：set -u 下空数组必须用安全展开写法
  if [[ $HAS_MUJIK -eq 1 ]]; then
    mujik run --input "$WORK_WAV" --output "$out_dir" \
      ${preset_args[@]+"${preset_args[@]}"} \
      ${ws_args[@]+"${ws_args[@]}"} || {
      echo "[demo] ⚠ run 失败（preset=${preset:-default}，继续）"
    }
  else
    python3 -m mujik.cli run --input "$WORK_WAV" --output "$out_dir" \
      ${preset_args[@]+"${preset_args[@]}"} \
      ${ws_args[@]+"${ws_args[@]}"} || {
      echo "[demo] ⚠ run 失败（preset=${preset:-default}，继续）"
    }
  fi
  if [[ $HAS_VEROVIO -eq 1 ]] && [[ -f "$out_dir/score.musicxml" ]]; then
    mujik render --input "$out_dir/score.musicxml" --output "$out_dir/score.pdf" --pdf \
      || echo "[demo] ⚠ render pdf 失败"
  fi
}

for p in "${PRESETS[@]}"; do
  run_once "$p"
done

# --- 6. 汇总报告 ---
python3 "$REPO_ROOT/scripts/_demo_report.py" "$OUT_ROOT" > "$OUT_ROOT/demo_report.md" \
  || echo "[demo] ⚠ report 失败"

# 清理临时裁剪
[[ "$WORK_WAV" != "$INPUT_WAV" ]] && [[ -f "$WORK_WAV" ]] && rm -f "$WORK_WAV"

echo
echo "✅ done → $SONG_DIR/"

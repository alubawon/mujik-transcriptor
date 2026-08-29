#!/usr/bin/env bash
# scripts/run_demo.sh — 一键跑 pop/jazz/metal 三 preset 对比 demo
# 默认用仓库自带 buhee/buhee.mp3；$1 可覆盖为自己的 wav
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- 0. 默认 + 覆盖 ---
DEFAULT_WAV="$REPO_ROOT/buhee/buhee.mp3"
INPUT_WAV="${1:-$DEFAULT_WAV}"
DURATION_ARG="${2:-}"
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

# --- 2. 探测时长（soundfile → ffprobe → 跳过）---
DUR="?"
if python3 -c "import soundfile, sys; sys.stdout.write(f'{soundfile.info(\"$INPUT_WAV\").duration:.1f}')" 2>/dev/null; then
  DUR=$(python3 -c "import soundfile; print(f'{soundfile.info(\"$INPUT_WAV\").duration:.1f}')" 2>/dev/null)
elif command -v ffprobe >/dev/null 2>&1; then
  DUR=$(ffprobe -v error -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "$INPUT_WAV" 2>/dev/null || echo "?")
fi
echo "[demo] input: $INPUT_WAV (${DUR}s)"

# --- 3. 可选：按 DURATION_ARG 裁剪 ---
WORK_WAV="$INPUT_WAV"
if [[ -n "$DURATION_ARG" ]]; then
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[demo] ⚠ ffmpeg 未装，跳过裁剪（用原文件）"
  else
    WORK_WAV="$OUT_ROOT/_trimmed_${RANDOM}.wav"
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

# --- 5. 跑三 preset ---
run_preset() {
  local preset=$1
  local out_dir="$OUT_ROOT/$preset"
  mkdir -p "$out_dir"
  echo
  echo "── preset=$preset ──"
  if [[ $HAS_MUJIK -eq 1 ]]; then
    mujik run --input "$WORK_WAV" --output "$out_dir" --preset "$preset" || {
      echo "[demo] ⚠ preset=$preset 失败（继续）"
    }
  else
    python3 -m mujik.cli run --input "$WORK_WAV" --output "$out_dir" --preset "$preset" || {
      echo "[demo] ⚠ preset=$preset 失败（继续）"
    }
  fi
  if [[ $HAS_VEROVIO -eq 1 ]] && [[ -f "$out_dir/score.musicxml" ]]; then
    mujik render --input "$out_dir/score.musicxml" --output "$out_dir/score.pdf" --pdf \
      || echo "[demo] ⚠ render pdf 失败"
  fi
}

run_preset pop
run_preset jazz
run_preset metal

# --- 6. 汇总报告 ---
python3 "$REPO_ROOT/scripts/_demo_report.py" "$OUT_ROOT" > "$OUT_ROOT/demo_report.md" \
  || echo "[demo] ⚠ report 失败"

# 清理临时裁剪
[[ "$WORK_WAV" != "$INPUT_WAV" ]] && [[ -f "$WORK_WAV" ]] && rm -f "$WORK_WAV"

echo
echo "✅ done → $OUT_ROOT/{pop,jazz,metal}/ + demo_report.md"

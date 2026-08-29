#!/usr/bin/env bash
# scripts/run_demo.sh — 一键跑 pop/jazz/metal 三 preset 对比 demo
#
# 用法（要求真实 wav 文件）：
#   ./scripts/run_demo.sh path/to/real_song.wav
#   ./scripts/run_demo.sh path/to/real_song.wav 30      # 可选：自定义时长
#
# 输出：
#   demo_out/
#   ├── pop/{project.mid, score.pdf, chords.json, ...}
#   ├── jazz/{...}
#   ├── metal/{...}
#   └── demo_report.md
#
# 注意：本脚本不接受默认 wav；必须显式提供真实音频（demo 不应跑合成数据，
# 否则对比 pop/jazz/metal 三个 preset 没有任何实际意义）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- 0. 参数解析（强制要求 input.wav）---
if [[ $# -lt 1 ]]; then
  echo "[demo] ❌ 必须提供真实音频 wav 路径作为第一个参数"
  echo ""
  echo "用法："
  echo "  $0 <input.wav>           # 三 preset 对比"
  echo "  $0 <input.wav> <duration_sec>  # 自定义裁剪时长（可选）"
  echo ""
  echo "示例："
  echo "  $0 ~/Music/pop_song.wav"
  echo "  $0 ~/Music/jazz_take.wav 45"
  echo ""
  echo "为什么强制要求真实 wav："
  echo "  pop/jazz/metal 三 preset 差异只在真实音乐上才可见；"
  echo "  跑仓库自带的合成正弦波对所有 preset 输出一致，没有 demo 价值。"
  exit 2
fi

INPUT_WAV="$1"
DURATION_ARG="${2:-}"
OUT_ROOT="$REPO_ROOT/demo_out"

# --- 1. 输入校验 ---
if [[ ! -f "$INPUT_WAV" ]]; then
  echo "[demo] ❌ input not found: $INPUT_WAV"
  echo "[demo] 提示：请检查路径是否正确，或用绝对路径"
  exit 1
fi

# 检查是 wav / flac / mp3 之一
INPUT_EXT="${INPUT_WAV##*.}"
case "${INPUT_EXT,,}" in
  wav|flac|mp3|ogg|m4a) ;;
  *)
    echo "[demo] ⚠ 未知格式: .$INPUT_EXT（期望 wav/flac/mp3/ogg/m4a）"
    ;;
esac

# ffprobe 可选：探测时长
DUR="?"
if command -v ffprobe >/dev/null 2>&1; then
  DUR=$(ffprobe -v error -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "$INPUT_WAV" 2>/dev/null || echo "?")
  DUR=${DUR%.*}
fi
echo "[demo] input: $INPUT_WAV (${DUR}s)"

# 可选：按 DURATION_ARG 裁剪
WORK_WAV="$INPUT_WAV"
if [[ -n "$DURATION_ARG" ]]; then
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[demo] ⚠ ffmpeg 未装，--duration 裁剪不可用；使用原文件"
  else
    WORK_WAV="$OUT_ROOT/_trimmed_${RANDOM}.wav"
    mkdir -p "$OUT_ROOT"
    ffmpeg -y -i "$INPUT_WAV" -t "$DURATION_ARG" -ar 44100 -ac 2 \
      "$WORK_WAV" >/dev/null 2>&1 || {
      echo "[demo] ❌ ffmpeg 裁剪失败，退出"
      exit 1
    }
    echo "[demo] trimmed to ${DURATION_ARG}s → $WORK_WAV"
  fi
fi

mkdir -p "$OUT_ROOT"

# --- 2. 环境探测 ---
HAS_MUJIK=0
HAS_DEMUCS=0
HAS_MADMOM=0
HAS_VEROVIO=0
HAS_FFMPEG=0

command -v mujik >/dev/null 2>&1 && HAS_MUJIK=1
command -v ffmpeg >/dev/null 2>&1 && HAS_FFMPEG=1
command -v verovio >/dev/null 2>&1 && HAS_VEROVIO=1
python3 -c "import demucs" 2>/dev/null && HAS_DEMUCS=1
python3 -c "import madmom" 2>/dev/null && HAS_MADMOM=1

echo "[demo] mujik=$HAS_MUJIK  ffmpeg=$HAS_FFMPEG  demucs=$HAS_DEMUCS  madmom=$HAS_MADMOM  verovio=$HAS_VEROVIO"

if [[ $HAS_MUJIK -eq 0 ]]; then
  echo "[demo] ⚠ mujik CLI 未安装；将用 python3 -m mujik.cli（需安装 mujik-transcriptor 包）"
fi
if [[ $HAS_DEMUCS -eq 0 ]]; then
  echo "[demo] ⚠ demucs 未安装；Demucs 阶段会失败（已 try/except 跳过）"
fi
if [[ $HAS_MADMOM -eq 0 ]]; then
  echo "[demo] ⚠ madmom 未安装；rhythm + chord（madmom backend）阶段会失败（已 try/except 跳过）"
fi
if [[ $HAS_VEROVIO -eq 0 ]]; then
  echo "[demo] ⚠ verovio 未安装；PDF 渲染会跳过（仍有 MusicXML 产物）"
fi

# --- 3. 跑三 preset ---
run_preset() {
  local preset=$1
  local out_dir="$OUT_ROOT/$preset"
  mkdir -p "$out_dir"
  echo
  echo "============================================================"
  echo "[demo] running preset=$preset → $out_dir"
  echo "============================================================"

  if [[ $HAS_MUJIK -eq 1 ]]; then
    mujik run --input "$WORK_WAV" --output "$out_dir" --preset "$preset" || {
      echo "[demo] ⚠ preset=$preset 失败，详见上方 traceback；继续下一 preset"
      return 1
    }
  else
    python3 -m mujik.cli run --input "$WORK_WAV" --output "$out_dir" --preset "$preset" || {
      echo "[demo] ⚠ preset=$preset 失败"
      return 1
    }
  fi

  # 渲染 PDF（若 verovio 可用 + score.musicxml 存在）
  if [[ $HAS_VEROVIO -eq 1 ]] && [[ -f "$out_dir/score.musicxml" ]]; then
    mujik render --input "$out_dir/score.musicxml" --output "$out_dir/score.pdf" --pdf \
      || echo "[demo] ⚠ render pdf 失败（继续）"
  else
    echo "[demo] render 跳过（verovio=$HAS_VEROVIO, musicxml 存在=$([ -f "$out_dir/score.musicxml" ] && echo y || echo n)）"
  fi
}

run_preset pop
run_preset jazz
run_preset metal

# --- 4. 出汇总报告 ---
echo
echo "[demo] generating summary report → $OUT_ROOT/demo_report.md"
python3 "$REPO_ROOT/scripts/_demo_report.py" "$OUT_ROOT" > "$OUT_ROOT/demo_report.md" \
  || echo "[demo] ⚠ report generation 失败"

# 清理临时裁剪文件
if [[ "$WORK_WAV" != "$INPUT_WAV" ]] && [[ -f "$WORK_WAV" ]]; then
  rm -f "$WORK_WAV"
fi

echo
echo "============================================================"
echo "[demo] ✅ done"
echo "  out dir:    $OUT_ROOT"
echo "  report:     $OUT_ROOT/demo_report.md"
echo "  per-preset: $OUT_ROOT/{pop,jazz,metal}/"
echo "============================================================"

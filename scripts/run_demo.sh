#!/usr/bin/env bash
# scripts/run_demo.sh — 一键跑 pop/jazz/metal 三 preset 对比 demo
# 用法：
#   ./scripts/run_demo.sh              # 用仓库自带 synthetic_5s.wav
#   ./scripts/run_demo.sh song.wav     # 用自己的 wav
#   ./scripts/run_demo.sh song.wav 30  # 自定义时长
#
# 输出：
#   demo_out/
#   ├── pop/{project.mid, score.pdf, chords.json, ...}
#   ├── jazz/{...}
#   ├── metal/{...}
#   └── report.md
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- 0. 参数解析 ---
INPUT_WAV="${1:-$REPO_ROOT/tests/fixtures/synthetic_5s.wav}"
DURATION_ARG="${2:-}"
OUT_ROOT="$REPO_ROOT/demo_out"
mkdir -p "$OUT_ROOT"

# --- 1. 输入准备 ---
if [[ ! -f "$INPUT_WAV" ]]; then
  echo "[demo] input not found: $INPUT_WAV"
  echo "[demo] 用法: $0 [input.wav] [duration_sec]"
  echo "[demo] 或者直接回车，跑仓库自带 synthetic_5s.wav"
  exit 1
fi

echo "[demo] input: $INPUT_WAV"
DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$INPUT_WAV" 2>/dev/null || echo "5.0")
DUR=${DUR%.*}
echo "[demo] duration: ${DUR}s"

# --- 2. 环境探测 ---
HAS_MUJIK=0
HAS_DEMUCS=0
HAS_MADMOM=0
HAS_VEROVIO=0

if command -v mujik >/dev/null 2>&1; then
  HAS_MUJIK=1
fi
if python -c "import demucs" 2>/dev/null; then
  HAS_DEMUCS=1
fi
if python -c "import madmom" 2>/dev/null; then
  HAS_MADMOM=1
fi
if command -v verovio >/dev/null 2>&1; then
  HAS_VEROVIO=1
fi

echo "[demo] mujik=$HAS_MUJIK  demucs=$HAS_DEMUCS  madmom=$HAS_MADMOM  verovio=$HAS_VEROVIO"

if [[ $HAS_MUJIK -eq 0 ]]; then
  echo "[demo] ⚠ mujik CLI 未安装；改用 python -m mujik.pipeline"
fi
if [[ $HAS_DEMUCS -eq 0 ]]; then
  echo "[demo] ⚠ demucs 未安装；demucs 阶段会被 mock 跳过"
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
    mujik run --input "$INPUT_WAV" --output "$out_dir" --preset "$preset" || {
      echo "[demo] preset=$preset failed, see logs above"
      return 1
    }
  else
    # fallback: python -m
    python -m mujik.cli run --input "$INPUT_WAV" --output "$out_dir" --preset "$preset" || {
      echo "[demo] preset=$preset failed"
      return 1
    }
  fi

  # 渲染 PDF（若 verovio 可用 + score.musicxml 存在）
  if [[ $HAS_VEROVIO -eq 1 ]] && [[ -f "$out_dir/score.musicxml" ]]; then
    mujik render --input "$out_dir/score.musicxml" --output "$out_dir/score.pdf" --pdf \
      || echo "[demo] render pdf failed (continuing)"
  else
    echo "[demo] render skipped (no verovio or no musicxml)"
  fi
}

run_preset pop
run_preset jazz
run_preset metal

# --- 4. 出汇总报告 ---
echo
echo "[demo] generating summary report → $OUT_ROOT/report.md"
python "$REPO_ROOT/scripts/_demo_report.py" "$OUT_ROOT" > "$OUT_ROOT/report.md" \
  || echo "[demo] report generation failed"

echo
echo "============================================================"
echo "[demo] ✅ done"
echo "  out dir: $OUT_ROOT"
echo "  report:  $OUT_ROOT/report.md"
echo "============================================================"

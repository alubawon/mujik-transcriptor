#!/usr/bin/env bash
# scripts/run_demo.sh — 一键 demo：音频 → MIDI + 乐谱
#
# 用法（命令即教程）：
#   ./scripts/run_demo.sh                            # 无参 = 三组合 showcase（一键全跑）：
#                                                    #   demo/buhee.mp3  × jazz
#                                                    #   demo/moon.mp3   × metal
#                                                    #   demo/dança.mp3  × pop
#   ./scripts/run_demo.sh path/to/your_song.wav      # 自己的音频（默认配置）
#   ./scripts/run_demo.sh demo/buhee.mp3 30          # 只跑前 30 秒（需 ffmpeg）
#   ./scripts/run_demo.sh demo/buhee.mp3 "" pop      # 显式指定 preset（默认配置即 pop）
#
# 多 preset 对比（opt-in，需显式传入单个音频，开发/评测用）：
#   MUJIK_DEMO_PRESETS="pop,jazz,metal" ./scripts/run_demo.sh demo/buhee.mp3
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
# 无参默认 = 三组合 showcase：每首曲子用「最像它风格」的 preset 跑一遍，
# 同时演示 --preset 用法。缺失的曲子跳过（警告），全部缺失才报错退出。
INPUT_WAV="${1:-}"
DURATION_ARG="${2:-}"
PRESET_ARG="${3:-}"
OUT_ROOT="$REPO_ROOT/demo_out"
mkdir -p "$OUT_ROOT"

# 组装 run 列表：每项 "input|preset"
RUNS=()
SHOWCASE_SKIPPED=0
if [[ -n "$INPUT_WAV" ]]; then
  if [[ ! -f "$INPUT_WAV" ]]; then
    echo "[demo] ❌ input not found: $INPUT_WAV"
    exit 1
  fi
  RUNS+=("$INPUT_WAV|$PRESET_ARG")
else
  if [[ -n "${MUJIK_DEMO_PRESETS:-}" ]]; then
    echo "[demo] ⚠ MUJIK_DEMO_PRESETS 需显式传入单个音频才生效（无参 showcase 按曲选 preset），已忽略"
  fi
  for entry in "demo/buhee.mp3|jazz" "demo/moon.mp3|metal" "demo/dança.mp3|pop"; do
    local_input="${entry%%|*}"
    if [[ -f "$REPO_ROOT/$local_input" ]]; then
      RUNS+=("$entry")
    else
      echo "[demo] ⚠ skip showcase 组合（文件不存在）：$local_input"
      SHOWCASE_SKIPPED=1
    fi
  done
  if [[ ${#RUNS[@]} -eq 0 ]]; then
    echo "[demo] ❌ showcase 曲目全部缺失，请把音频放到 demo/ 或显式传入: $0 your_song.wav"
    exit 1
  fi
fi

# --- 环境探测（整脚本一次）---
HAS_MUJIK=0
command -v mujik >/dev/null 2>&1 && HAS_MUJIK=1
# verovio 能力 = python 包（verovio 无独立 CLI 可执行；SVG→PDF 走 mujik render 内部管线）
HAS_VEROVIO=0; python3 -c "import verovio" 2>/dev/null && HAS_VEROVIO=1
python3 -c "import demucs" 2>/dev/null && HAS_DEMUCS=1 || HAS_DEMUCS=0
python3 -c "import madmom" 2>/dev/null && HAS_MADMOM=1 || HAS_MADMOM=0
echo "[demo] mujik=$HAS_MUJIK  demucs=$HAS_DEMUCS  madmom=$HAS_MADMOM  verovio=$HAS_VEROVIO"

run_once() {
  local input="$1" preset="$2"
  local song song_dir
  song="$(basename "$input")"; song="${song%.*}"
  song_dir="$OUT_ROOT/$song"
  mkdir -p "$song_dir"

  # 探测时长（soundfile → ffprobe → 跳过）
  local dur="?"
  if python3 -c "import soundfile, sys; sys.stdout.write(f'{soundfile.info(\"$input\").duration:.1f}')" 2>/dev/null; then
    dur=$(python3 -c "import soundfile; print(f'{soundfile.info(\"$input\").duration:.1f}')" 2>/dev/null)
  elif command -v ffprobe >/dev/null 2>&1; then
    dur=$(ffprobe -v error -show_entries format=duration \
          -of default=noprint_wrappers=1:nokey=1 "$input" 2>/dev/null || echo "?")
  fi
  echo
  echo "━━━ $song  [preset=${preset:-default}]  input: $input (${dur}s) → ${song_dir#$REPO_ROOT/}/ ━━━"

  # 可选：按 DURATION_ARG 裁剪（确定性命名，同曲同长度可复用）
  local work_wav="$input"
  if [[ -n "$DURATION_ARG" ]]; then
    if ! command -v ffmpeg >/dev/null 2>&1; then
      echo "[demo] ⚠ ffmpeg 未装，跳过裁剪（用原文件）"
    else
      work_wav="$OUT_ROOT/_trimmed_${song}_${DURATION_ARG}s.wav"
      ffmpeg -y -i "$input" -t "$DURATION_ARG" -ar 44100 -ac 2 \
        "$work_wav" >/dev/null 2>&1 || {
        echo "[demo] ❌ ffmpeg 裁剪失败"; return 1;
      }
      echo "[demo] trimmed → ${DURATION_ARG}s"
    fi
  fi

  # preset 路由：MUJIK_DEMO_PRESETS（对比模式，仅显式单输入）> 显式 $3 > 默认（不传 flag，即 pop 配置）
  local comparison=0
  local presets=()
  if [[ -n "$INPUT_WAV" ]] && [[ -n "${MUJIK_DEMO_PRESETS:-}" ]]; then
    comparison=1
    IFS=',' read -r -a presets <<< "$MUJIK_DEMO_PRESETS"
  else
    presets=("$preset")
  fi

  for p in "${presets[@]}"; do
    local out_dir preset_args=() ws_args=()
    if [[ $comparison -eq 1 ]]; then
      out_dir="$song_dir/$p"          # 对比模式：按 preset 分目录，共享 ws
      ws_args=(--workspace "$song_dir/ws")
    else
      out_dir="$song_dir"             # 单 run：最终产物直接在 曲名/ 下
    fi
    [[ -n "$p" ]] && preset_args=(--preset "$p")

    echo "── run → ${out_dir#$REPO_ROOT/} ${p:+[preset=$p]} ──"
    mkdir -p "$out_dir"
    # macOS bash 3.2 兼容：set -u 下空数组必须用安全展开写法
    if [[ $HAS_MUJIK -eq 1 ]]; then
      mujik run --input "$work_wav" --output "$out_dir" \
        ${preset_args[@]+"${preset_args[@]}"} \
        ${ws_args[@]+"${ws_args[@]}"} || {
        echo "[demo] ⚠ run 失败（preset=${p:-default}，继续）"
      }
    else
      python3 -m mujik.cli run --input "$work_wav" --output "$out_dir" \
        ${preset_args[@]+"${preset_args[@]}"} \
        ${ws_args[@]+"${ws_args[@]}"} || {
        echo "[demo] ⚠ run 失败（preset=${p:-default}，继续）"
      }
    fi
    if [[ $HAS_VEROVIO -eq 1 ]] && [[ -f "$out_dir/score.musicxml" ]]; then
      mujik render --input "$out_dir/score.musicxml" --output "$out_dir/score.pdf" --pdf \
        || echo "[demo] ⚠ render pdf 失败"
    fi
  done

  # 清理本次裁剪
  if [[ "$work_wav" != "$input" ]] && [[ -f "$work_wav" ]]; then
    rm -f "$work_wav"
  fi
}

for entry in "${RUNS[@]}"; do
  run_once "${entry%%|*}" "${entry##*|}" || echo "[demo] ⚠ $entry 跑失败，继续"
done

# --- 汇总报告 ---
python3 "$REPO_ROOT/scripts/_demo_report.py" "$OUT_ROOT" > "$OUT_ROOT/demo_report.md" \
  || echo "[demo] ⚠ report 失败"

echo
echo "✅ done → $OUT_ROOT/（报告: $OUT_ROOT/demo_report.md）"

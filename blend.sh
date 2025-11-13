#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: blend.sh [options]

Blend transformer checkpoints at configurable ratios and (optionally) run the
evaluation loop on CPU. Multiple --ratio flags may be provided to sweep
different blends in one pass; ratio formats such as "0.6,0.4" or "60:40" are
accepted.

Options:
  --base PATH          Path to the baseline checkpoint.
  --finetune PATH      Path to the finetuned/SWA checkpoint.
  --prefix NAME        Prefix for blended checkpoint filenames (default: selective_swa_epoch09_12).
  --ratio SPEC         Blend ratio (e.g. 0.5,0.5 or 60:40). Repeat for a sweep.
  --ratios STRING      Comma-separated list of ratio specs (shorthand for repeating --ratio).
  --no-eval            Skip evaluation after blending.
  --log-dir DIR        Directory for evaluation logs (default: logs/blend_evals).
  --output-dir DIR     Directory for blended checkpoints (defaults to finetune checkpoint dir).
  --conda-env NAME     Conda environment to activate (default: pocknet_env).
  --help               Show this help message and exit.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
CONDA_ENV_NAME=${CONDA_ENV_NAME:-pocknet_env}

CKPT_BASE_DEFAULT=logs/fusion_transformer_aggressive_oct17/runs/2025-10-21_12-48-21/checkpoints/epoch_09-val_auprc_0.3132.ckpt
CKPT_FINETUNE_DEFAULT=logs/fusion_transformer_aggressive_oct17/runs/2025-10-21_20-37-35/checkpoints/finetune_epoch_12-val_auprc_0.2985.ckpt

CKPT_BASE="$CKPT_BASE_DEFAULT"
CKPT_FINETUNE="$CKPT_FINETUNE_DEFAULT"
BLEND_PREFIX="selective_swa_epoch09_12"
OUTPUT_DIR=""
LOG_DIR="logs/blend_evals"
DO_EVAL=true
declare -a RATIO_SPECS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --base"; exit 1; }
      CKPT_BASE="$1"
      shift
      ;;
    --finetune)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --finetune"; exit 1; }
      CKPT_FINETUNE="$1"
      shift
      ;;
    --prefix)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --prefix"; exit 1; }
      BLEND_PREFIX="$1"
      shift
      ;;
    --ratio)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --ratio"; exit 1; }
      RATIO_SPECS+=("$1")
      shift
      ;;
    --ratios)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --ratios"; exit 1; }
      ratio_arg="${1//;/ }"
      read -r -a parsed_ratios <<< "$ratio_arg"
      for spec in "${parsed_ratios[@]}"; do
        if [[ -n "${spec// }" ]]; then
          RATIO_SPECS+=("$spec")
        fi
      done
      shift
      ;;
    --no-eval)
      DO_EVAL=false
      shift
      ;;
    --log-dir)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --log-dir"; exit 1; }
      LOG_DIR="$1"
      shift
      ;;
    --output-dir)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --output-dir"; exit 1; }
      OUTPUT_DIR="$1"
      shift
      ;;
    --conda-env)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --conda-env"; exit 1; }
      CONDA_ENV_NAME="$1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ${#RATIO_SPECS[@]} -eq 0 ]]; then
  RATIO_SPECS=("0.5,0.5")
fi

if [[ ! -f "$CKPT_BASE" ]]; then
  echo "Baseline checkpoint not found: $CKPT_BASE" >&2
  exit 1
fi

if [[ ! -f "$CKPT_FINETUNE" ]]; then
  echo "Finetune checkpoint not found: $CKPT_FINETUNE" >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$(cd "$(dirname "$CKPT_FINETUNE")" && pwd)"
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)" >/dev/null 2>&1
  conda activate "$CONDA_ENV_NAME"
else
  echo "Conda not found in PATH" >&2
  exit 1
fi

cd "$ROOT_DIR"

parse_ratio() {
  local spec="$1"
  mapfile -t parsed < <(python - <<'PY' "$spec"
import sys
import math

raw = sys.argv[1]
if not raw.strip():
    raise SystemExit("Empty ratio specification")

separators = [",", "/", ":", " "]
parts = None
for sep in separators:
    if sep in raw:
        parts = [p for p in raw.split(sep) if p.strip()]
        break

if parts is None:
    parts = [raw.strip()]

values = []
for part in parts:
    try:
        val = float(part)
    except ValueError as err:
        raise SystemExit(f"Invalid float in ratio '{raw}': {part}") from err
    if val > 1.0:
        val /= 100.0
    values.append(val)

if len(values) < 2:
    raise SystemExit(f"Ratio '{raw}' must contain at least two weights")

total = sum(values)
if math.isclose(total, 0.0, abs_tol=1e-12):
    raise SystemExit(f"Ratio '{raw}' sums to zero")

norm = [v / total for v in values]
label = "w" + "_".join(f"{int(round(v * 100))}" for v in norm)
weights_line = " ".join(f"{v:.6f}" for v in norm)
print(weights_line)
print(label)
PY
)
  if [[ ${#parsed[@]} -lt 2 ]]; then
    echo "Failed to parse ratio '$spec'" >&2
    return 1
  fi
  echo "${parsed[0]}|${parsed[1]}"
}

summarize_metric() {
  local log_path="$1"
  local metric_line=""
  if command -v rg >/dev/null 2>&1; then
    metric_line=$(rg "test/auprc" "$log_path" | tail -n1 || true)
  else
    metric_line=$(grep "test/auprc" "$log_path" | tail -n1 || true)
  fi
  if [[ -n "$metric_line" ]]; then
    echo "    ${metric_line}"
  else
    echo "    test/auprc not found in ${log_path}"
  fi
}

for spec in "${RATIO_SPECS[@]}"; do
  parse_result="$(parse_ratio "$spec")" || exit 1
  weights_line="${parse_result%%|*}"
  label="${parse_result##*|}"
  read -r -a weights <<< "$weights_line"

  if [[ "$label" == "w50_50" ]]; then
    blend_path="${OUTPUT_DIR}/${BLEND_PREFIX}.ckpt"
    log_path="${LOG_DIR}/${BLEND_PREFIX}.log"
  else
    blend_path="${OUTPUT_DIR}/${BLEND_PREFIX}_${label}.ckpt"
    log_path="${LOG_DIR}/${BLEND_PREFIX}_${label}.log"
  fi

  echo "============================================================"
  echo "Blending ratio: ${spec} (normalized: ${weights[*]})"
  echo "Output checkpoint: ${blend_path}"

  python src/scripts/blend_checkpoints.py \
    --inputs "$CKPT_BASE" "$CKPT_FINETUNE" \
    --weights "${weights[@]}" \
    --output "$blend_path"

  if "$DO_EVAL"; then
    echo "Evaluating checkpoint -> ${log_path}"
    SHM_DIR="$(mktemp -d /dev/shm/pocknet_${USER}_XXXXXX 2>/dev/null || true)"
    if [[ -z "${SHM_DIR:-}" ]]; then
      FALLBACK_DIR="tmp/pocknet_shm_${USER}"
      mkdir -p "$FALLBACK_DIR"
      SHM_DIR="$(mktemp -d "${FALLBACK_DIR}/XXXXXX" 2>/dev/null || true)"
    fi
    cleanup_shm() {
      if [[ -n "${SHM_DIR:-}" && -d "${SHM_DIR:-}" ]]; then
        rm -rf "${SHM_DIR}"
      fi
      SHM_DIR=""
    }
    trap cleanup_shm EXIT

    if [[ -n "${SHM_DIR:-}" ]]; then
      echo "Using /dev/shm cache: ${SHM_DIR}"
      export POCKNET_SHM_DIR="$SHM_DIR"
    else
      echo "Warning: unable to allocate /dev/shm directory; proceeding without POCKNET_SHM_DIR"
      unset POCKNET_SHM_DIR
    fi

    set +e
    python src/train.py \
      experiment=fusion_transformer_aggressive_oct17 \
      train=false test=true \
      trainer.accelerator=cpu \
      trainer.devices=1 \
      trainer.strategy=auto \
      logger=csv \
      ckpt_path="$blend_path" \
      2>&1 | tee "$log_path"
    cmd_status=${PIPESTATUS[0]}
    set -e

    cleanup_shm
    trap - EXIT
    unset POCKNET_SHM_DIR

    if [[ $cmd_status -ne 0 ]]; then
      echo "Evaluation failed for ratio ${spec}" >&2
      exit "$cmd_status"
    fi

    summarize_metric "$log_path"
  fi
done

echo "============================================================"
echo "Done."

#!/bin/bash
# Evaluate a set of checkpoints with Lightning/Hydra and stream all results
# into a single log file for easy comparison.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: eval_checkpoints.sh [options] [hydra_overrides...]

Options:
  --dir PATH        Directory containing checkpoints to evaluate (required).
  --pattern GLOB    Filename pattern to match (default: selective_swa_epoch09_12*.ckpt).
  --experiment EXP  Hydra experiment config to use (default: fusion_transformer_aggressive_oct17).
  --log FILE        Path to the aggregate tee log (default: logs/blend_evals/eval_YYYYmmdd_HHMMSS.log).
  --stop-on-fail    Abort immediately if any evaluation fails (default: continue).
  --help            Show this message and exit.

Any extra positional arguments are forwarded to Hydra, e.g.
  eval_checkpoints.sh --dir ... trainer.devices=1 trainer.accelerator=gpu
EOF
}

CKPT_DIR=""
PATTERN="selective_swa_epoch09_12*.ckpt"
EXPERIMENT="fusion_transformer_aggressive_oct17"
LOG_FILE=""
STOP_ON_FAIL=false
declare -a EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --dir"; exit 1; }
      CKPT_DIR="$1"
      shift
      ;;
    --pattern)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --pattern"; exit 1; }
      PATTERN="$1"
      shift
      ;;
    --experiment)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --experiment"; exit 1; }
      EXPERIMENT="$1"
      shift
      ;;
    --log)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --log"; exit 1; }
      LOG_FILE="$1"
      shift
      ;;
    --stop-on-fail)
      STOP_ON_FAIL=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      EXTRA_OVERRIDES+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$CKPT_DIR" ]]; then
  echo "Error: --dir is required." >&2
  usage
  exit 1
fi

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "Checkpoint directory not found: $CKPT_DIR" >&2
  exit 1
fi

if [[ -z "$LOG_FILE" ]]; then
  mkdir -p logs/blend_evals
  LOG_FILE="logs/blend_evals/eval_$(date +%Y%m%d_%H%M%S).log"
else
  mkdir -p "$(dirname "$LOG_FILE")"
fi

shopt -s nullglob
mapfile -t CKPTS < <(cd "$CKPT_DIR" && ls $PATTERN 2>/dev/null | sort)
shopt -u nullglob

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "No checkpoints matching '$PATTERN' in $CKPT_DIR" >&2
  exit 1
fi

echo "Writing aggregated results to $LOG_FILE"
echo "Experiment: $EXPERIMENT" | tee -a "$LOG_FILE"
echo "Directory:  $CKPT_DIR" | tee -a "$LOG_FILE"
echo "Pattern:    $PATTERN" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

EXIT_STATUS=0
for ckpt in "${CKPTS[@]}"; do
  ckpt_path="${CKPT_DIR%/}/$ckpt"
  ts=$(date +%F_%H-%M-%S)
  echo "" | tee -a "$LOG_FILE"
  echo ">>> [$ts] Evaluating $(basename "$ckpt_path")" | tee -a "$LOG_FILE"
  echo "Command: python src/train.py experiment=$EXPERIMENT train=false test=true ckpt_path=$ckpt_path ${EXTRA_OVERRIDES[*]}" | tee -a "$LOG_FILE"

  set +e
  python src/train.py \
    "experiment=$EXPERIMENT" \
    train=false test=true \
    "ckpt_path=$ckpt_path" \
    "${EXTRA_OVERRIDES[@]}" \
    2>&1 | tee -a "$LOG_FILE"
  status=${PIPESTATUS[0]}
  set -e

  if [[ $status -ne 0 ]]; then
    echo "!!! Evaluation failed for $ckpt_path (exit $status)" | tee -a "$LOG_FILE"
    EXIT_STATUS=$status
    if [[ "$STOP_ON_FAIL" == true ]]; then
      break
    fi
  fi
done

exit "$EXIT_STATUS"

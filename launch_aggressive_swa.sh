#!/bin/bash
# Launch aggressive transformer training with an SWA finetuning phase.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: launch_aggressive_swa.sh [options] [hydra_overrides...]

Options:
  --ckpt PATH           Checkpoint to resume from (baseline, finetune, or blend).
  --swa-start EPOCH     Epoch to start SWA averaging (default: 20).
  --swa-length EPOCHS   Number of epochs in the SWA averaging window (default: 10).
  --swa-lr VALUE        Learning rate used during SWA phase (default: 0.0002).
  --max-epochs N        Trainer max epochs (default: 60).
  --anneal-strategy STR SWA annealing strategy (default: cos).
  --patience N          Early stopping patience (default: 1000).
  --session NAME        tmux session name (default: transformer_training_swa).
  --experiment NAME     Lightning/Hydra experiment config (default: fusion_transformer_aggressive or \$EXPERIMENT_OVERRIDE).
  --preset short        Short SWA window (start=26, length=6, max_epochs=36) with heavier pos_weight=45.
  --hard-pos PATH       JSON file containing hard-positive indices to replay.
  --hard-repeat N       Repeat factor for the hard-positive list (default: 3 in short preset).
  --no-tmux             Run inline instead of launching a tmux session.
  --help                Show this help message and exit.

Any additional positional arguments are forwarded to Hydra as overrides.
EOF
}

EXPERIMENT=${EXPERIMENT_OVERRIDE:-fusion_transformer_aggressive}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-p2rank_env}
SESSION_NAME="transformer_training_swa"
CKPT_PATH=""
SWA_START=20
SWA_LENGTH=10
SWA_LR=0.0002
MAX_EPOCHS=60
ANNEAL_STRATEGY="cos"
PATIENCE=1000
USE_TMUX=true
POS_WEIGHT_OVERRIDE=""

declare -a EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt|--resume)
      shift
      [[ $# -gt 0 ]] || { echo "Missing path for --ckpt"; exit 1; }
      CKPT_PATH="$1"
      shift
      ;;
    --swa-start)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --swa-start"; exit 1; }
      SWA_START="$1"
      shift
      ;;
    --swa-length|--anneal)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --swa-length"; exit 1; }
      SWA_LENGTH="$1"
      shift
      ;;
    --swa-lr)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --swa-lr"; exit 1; }
      SWA_LR="$1"
      shift
      ;;
    --max-epochs)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --max-epochs"; exit 1; }
      MAX_EPOCHS="$1"
      shift
      ;;
    --anneal-strategy)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --anneal-strategy"; exit 1; }
      ANNEAL_STRATEGY="$1"
      shift
      ;;
    --patience)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --patience"; exit 1; }
      PATIENCE="$1"
      shift
      ;;
    --session)
      shift
      [[ $# -gt 0 ]] || { echo "Missing name for --session"; exit 1; }
      SESSION_NAME="$1"
      shift
      ;;
    --experiment)
      shift
      [[ $# -gt 0 ]] || { echo "Missing name for --experiment"; exit 1; }
      EXPERIMENT="$1"
      shift
      ;;
    --preset)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --preset"; exit 1; }
      case "$1" in
        short)
          SWA_START=26
          SWA_LENGTH=6
          MAX_EPOCHS=36
          if [[ -z "$POS_WEIGHT_OVERRIDE" ]]; then
            POS_WEIGHT_OVERRIDE="45.0"
          fi
          EXTRA_OVERRIDES+=("data.hard_positive_repeat=3")
          ;;
        *)
          echo "Unknown preset: $1" >&2
          exit 1
          ;;
      esac
      shift
      ;;
    --hard-pos)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --hard-pos"; exit 1; }
      EXTRA_OVERRIDES+=("data.hard_positive_indices_path=$1")
      shift
      ;;
    --hard-repeat)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --hard-repeat"; exit 1; }
      EXTRA_OVERRIDES+=("data.hard_positive_repeat=$1")
      shift
      ;;
    --pos-weight)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --pos-weight"; exit 1; }
      POS_WEIGHT_OVERRIDE="$1"
      shift
      ;;
    --no-tmux)
      USE_TMUX=false
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

if [[ -n "$CKPT_PATH" && ! -f "$CKPT_PATH" ]]; then
  echo "❌ Provided checkpoint path not found: $CKPT_PATH" >&2
  exit 1
fi

if [[ "$USE_TMUX" != true && "$USE_TMUX" != false ]]; then
  echo "Invalid value for USE_TMUX: $USE_TMUX" >&2
  exit 1
fi

OVERRIDES=(
  "experiment=$EXPERIMENT"
  "trainer.max_epochs=$MAX_EPOCHS"
  "callbacks.early_stopping.patience=$PATIENCE"
  "+callbacks.swa._target_=lightning.pytorch.callbacks.StochasticWeightAveraging"
  "+callbacks.swa.swa_epoch_start=$SWA_START"
  "+callbacks.swa.annealing_epochs=$SWA_LENGTH"
  "+callbacks.swa.annealing_strategy=$ANNEAL_STRATEGY"
  "+callbacks.swa.swa_lrs=$SWA_LR"
)

if [[ -n "$CKPT_PATH" ]]; then
  OVERRIDES+=("ckpt_path=$CKPT_PATH")
fi

for extra in "${EXTRA_OVERRIDES[@]}"; do
  OVERRIDES+=("$extra")
done

if [[ -n "$POS_WEIGHT_OVERRIDE" ]]; then
  OVERRIDES+=("+model.pos_weight=$POS_WEIGHT_OVERRIDE")
  OVERRIDES+=("callbacks.imbalance_setup.manual_pos_weight=$POS_WEIGHT_OVERRIDE")
fi

LOG_PATH="training_aggressive_swa_$(date +%F_%H-%M-%S).log"

echo "=========================================="
echo "🚀 Aggressive Transformer + SWA Training"
echo "=========================================="
echo ""
echo "📊 Training Plan:"
echo "   • Experiment: ${EXPERIMENT}"
echo "   • Checkpoint: ${CKPT_PATH:-<fresh start>}"
echo "   • SWA start / length: ${SWA_START} / ${SWA_LENGTH}"
echo "   • Max epochs: ${MAX_EPOCHS}"
echo "   • SWA LR: ${SWA_LR}"
echo ""

echo "🧹 Stopping existing session (${SESSION_NAME})..."
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
sleep 2

echo "🔧 Activating conda environment..."
if command -v conda >/dev/null 2>&1; then
  set +u
  if [[ -f "$HOME/.bashrc" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/.bashrc" >/dev/null 2>&1 || true
  fi
  set -u
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)" >/dev/null 2>&1
  conda activate "$CONDA_ENV_NAME"
else
  echo "❌ Conda not found on PATH." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "📊 GPU Status:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || \
    echo "⚠️  nvidia-smi available but could not query devices."
else
  echo "⚠️  nvidia-smi not found on PATH."
fi

GPU_COUNT=$(python - <<'PY' 2>/dev/null || echo "0"
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
if [[ "$GPU_COUNT" =~ ^[0-9]+$ ]] && [[ "$GPU_COUNT" -gt 0 ]]; then
  GPU_MSG="torch.cuda reports ${GPU_COUNT} visible GPU(s)"
else
  GPU_MSG="torch.cuda could not detect GPUs (Lightning will use defaults)"
fi

TRAIN_CMD="python src/train.py"
for override in "${OVERRIDES[@]}"; do
  quoted_override=$(printf '%q' "$override")
  TRAIN_CMD+=" ${quoted_override}"
done
TRAIN_CMD+=" 2>&1 | tee $(printf '%q' "${LOG_PATH}")"

echo ""
echo "🚀 Launching training with SWA finetune..."
echo "   $GPU_MSG"
echo "   Log file: ${LOG_PATH}"
echo ""

RUN_CMD="cd \"$ROOT_DIR\" && export PROJECT_ROOT=\"$ROOT_DIR\" && $TRAIN_CMD"
USED_TMUX=0

if [[ "$USE_TMUX" == true ]] && command -v tmux >/dev/null 2>&1; then
  if tmux new-session -d -s "$SESSION_NAME" bash -c "$RUN_CMD"; then
    echo "✅ Training launched in tmux session '$SESSION_NAME'"
    USED_TMUX=1
  else
    echo "⚠️ tmux launch failed. Running training inline instead."
    bash -c "$RUN_CMD"
  fi
else
  if [[ "$USE_TMUX" == true ]]; then
    echo "⚠️ tmux not found. Running training inline."
  fi
  bash -c "$RUN_CMD"
fi

if [[ "$USED_TMUX" -eq 1 ]]; then
  echo "📊 Monitor with:"
  echo "   tmux attach -t $SESSION_NAME"
  echo "   tail -f ${LOG_PATH}"
  echo ""
else
  echo "📊 Monitor with: tail -f ${LOG_PATH}"
  echo ""
fi
echo "🔍 W&B Dashboard:"
echo "   https://wandb.ai/max-hageneder-johannes-kepler-universit-t-linz/fusion_pocknet_thesis"

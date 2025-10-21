#!/bin/bash
# Launch aggressive transformer training with an SWA finetuning phase.

set -e

SESSION_NAME="transformer_training_swa"
CKPT_PATH=${1:-}
if [[ -n "$CKPT_PATH" && ! -f "$CKPT_PATH" ]]; then
  echo "❌ Provided checkpoint path not found: $CKPT_PATH" >&2
  exit 1
fi

OVERRIDES=(
  experiment=fusion_transformer_aggressive
  trainer.max_epochs=60
  callbacks.early_stopping.patience=1000
  +callbacks.swa._target_=lightning.pytorch.callbacks.StochasticWeightAveraging
  +callbacks.swa.swa_epoch_start=20
  +callbacks.swa.annealing_epochs=10
  +callbacks.swa.annealing_strategy=cos
  +callbacks.swa.swa_lrs=0.0002
)

if [[ -n "$CKPT_PATH" ]]; then
  OVERRIDES+=("ckpt_path=$CKPT_PATH")
fi

# Build command string with proper quoting
TRAIN_CMD="python src/train.py"
for override in "${OVERRIDES[@]}"; do
  TRAIN_CMD+=" \"${override}\""
done
TRAIN_CMD+=" 2>&1 | tee training_aggressive_swa_$(date +%F_%H-%M-%S).log"

echo "=========================================="
echo "🚀 Aggressive Transformer + SWA Training"
echo "=========================================="
echo ""
echo "📊 Training Plan:"
echo "   • Base experiment: fusion_transformer_aggressive"
echo "   • SWA starts at epoch 20 with a 10-epoch cosine window"
echo "   • Aux head weight anneals 0.07 → 0.015; distance weighting enabled"
echo "   • Gate penalty relaxes 0.16 → 0.08, dropout target 0.60, context scale 0.45 → 0.60"
echo "   • Trainer capped at 60 epochs (no early stop)"
echo ""

echo "🧹 Stopping existing session ($SESSION_NAME)..."
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
sleep 2

echo "🔧 Activating conda environment..."
source ~/.bashrc
conda activate p2rank_env

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

echo ""
echo "🚀 Launching training with SWA finetune..."
echo "   $GPU_MSG"
echo "   Log file prefix: training_aggressive_swa_(timestamp).log"
echo ""

# Build run command
RUN_CMD="cd \"$ROOT_DIR\" && export PROJECT_ROOT=\"$ROOT_DIR\" && $TRAIN_CMD"
USED_TMUX=0

if command -v tmux >/dev/null 2>&1; then
  if tmux new-session -d -s "$SESSION_NAME" bash -c "$RUN_CMD"; then
    echo "✅ Training launched in tmux session '$SESSION_NAME'"
    USED_TMUX=1
  else
    echo "⚠️ tmux launch failed. Running training inline instead."
    bash -c "$RUN_CMD"
  fi
else
  echo "⚠️ tmux not found. Running training inline."
  bash -c "$RUN_CMD"
fi

if [[ "$USED_TMUX" -eq 1 ]]; then
  echo "📊 Monitor with:"
  echo "   tmux attach -t $SESSION_NAME"
  echo "   tail -f training_aggressive_swa_*.log"
  echo ""
else
  echo "📊 Monitor with: tail -f training_aggressive_swa_*.log"
  echo ""
fi
echo "🔍 W&B Dashboard:"
echo "   https://wandb.ai/max-hageneder-johannes-kepler-universit-t-linz/fusion_pocknet_thesis"

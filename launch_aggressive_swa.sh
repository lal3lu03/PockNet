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
echo "   • Aux head weight anneals 0.10 → 0.02; distance weighting enabled"
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
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | \
    awk -F, '{printf "GPU %s: %s (%.0f GB free / %.0f GB total)\n", $1, $2, $4/1024, $3/1024}'

export CUDA_VISIBLE_DEVICES=1,3,4

echo ""
echo "🚀 Launching training with SWA finetune..."
echo "   Log file prefix: training_aggressive_swa_(timestamp).log"
echo ""

tmux new-session -d -s "$SESSION_NAME" bash -c "cd \"$ROOT_DIR\" && export PROJECT_ROOT=\"$ROOT_DIR\" && export CUDA_VISIBLE_DEVICES=1,3,4 && $TRAIN_CMD"

echo "✅ Training launched in tmux session '$SESSION_NAME'"
echo ""
echo "📊 Monitor with:"
echo "   tmux attach -t $SESSION_NAME"
echo "   tail -f training_aggressive_swa_*.log"
echo ""
echo "🔍 W&B Dashboard:"
echo "   https://wandb.ai/max-hageneder-johannes-kepler-universit-t-linz/fusion_pocknet_thesis"

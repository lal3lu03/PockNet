#!/bin/bash
# Launch Aggressive Transformer Training
# Quick Win Strategy: More capacity, lower LR, force neighbor learning
# Target: 0.27-0.30 AUPRC (up from 0.24)

set -e

EXPERIMENT=${EXPERIMENT_OVERRIDE:-fusion_transformer_aggressive}
CMD_OVERRIDES=("$@")

echo "=========================================="
echo "Aggressive Transformer k-NN Training"
echo "=========================================="
echo ""
echo "Configuration Changes (Professional Optimization):"
echo "   - Attention heads: 6 with 2400-dim projection (balanced per-head capacity)"
echo "   - Transformer layers: 2 (same, avoid parameter explosion)"
echo "   - Feedforward dim: 1280 (was 1024) [+25%]"
echo "   - Max LR: 5e-4 (was 1e-3) [conservative for larger batch]"
echo "   - Batch size: 640 (was 512) [more positives per GPU]"
echo "   - Accumulate batches: 1 (was 2) [effective batch = 640 x GPUs]"
echo "   - Max epochs: 80 (unchanged) [peak typically ~epoch 25]"
echo "   - Patience: 15 (unchanged) [aggressive early stopping]"
echo "   - pct_start: 0.2 (was 0.3) [faster warmup for larger batch]"
echo "   - Modality dropout: 0.1 (was 0.0) [force neighbor use]"
echo "   - Residual gate: penalty 0.16->0.08 toward alpha ~0.75, dropout 25%->10%, temperature 2.0"
echo "   - Context scale anneals 0.45 -> 0.60 over the first 15 epochs"
echo "   - Neighbour-only aux head (0.07->0.015 over 10 epochs) keeps attention gradients alive"
echo "   - Distance clamp: 30 Angstrom (was 0) [better bias learning]"
echo "   - Validation logging: per-head attention entropy and normalized context norms"
echo ""
echo "Target: 0.27-0.29 AUPRC (scientific optimization)"
echo "Expected: +3-5% over baseline, faster convergence, less overfit"
echo ""

# Kill any existing training
echo "Stopping existing training..."
tmux kill-session -t transformer_training 2>/dev/null || true
sleep 2

# Activate environment
echo "Activating conda environment..."
source ~/.bashrc
conda activate pocknet_env

# Resolve project root (directory containing this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check GPUs (best effort)
echo ""
echo "GPU Status (best effort):"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || \
    echo "Warning: nvidia-smi available but could not query devices."
else
  echo "Warning: nvidia-smi not found on PATH."
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
echo "Launching aggressive transformer training..."
echo "   Experiment: ${EXPERIMENT}"
echo "   $GPU_MSG"
echo "   Log prefix: training_aggressive_(timestamp).log"
echo ""

# Build run command
RUN_CMD="cd \"$ROOT_DIR\" && export PROJECT_ROOT=\"$ROOT_DIR\" && python src/train.py experiment=${EXPERIMENT}"
for override in "${CMD_OVERRIDES[@]}"; do
  RUN_CMD+=" \"${override}\""
done
RUN_CMD+=" 2>&1 | tee training_aggressive_\$(date +%F_%H-%M-%S).log"
USED_TMUX=0

# Launch in tmux when available, otherwise run inline
if command -v tmux >/dev/null 2>&1; then
  if tmux new-session -d -s transformer_training bash -c "$RUN_CMD"; then
    echo "Training launched in tmux session 'transformer_training'"
    USED_TMUX=1
  else
    echo "Warning: tmux launch failed. Running training inline instead."
    bash -c "$RUN_CMD"
  fi
else
  echo "Warning: tmux not found. Running training inline."
  bash -c "$RUN_CMD"
fi
echo ""
if [[ "$USED_TMUX" -eq 1 ]]; then
  echo "Monitor with:"
  echo "   tmux attach -t transformer_training"
  echo "   tail -f training_aggressive_*.log"
  echo ""
else
  echo "Monitor with: tail -f training_aggressive_*.log"
  echo ""
fi
echo "W&B Dashboard:"
echo "   https://wandb.ai/max-hageneder-johannes-kepler-universit-t-linz/fusion_pocknet_thesis"
echo ""
echo "Expected training time: ~18-24 hours (<=80 epochs)"
echo "Look for AUPRC > 0.27 by epoch 12-20"

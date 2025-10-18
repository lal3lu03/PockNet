#!/bin/bash
# Launch Aggressive Transformer Training
# Quick Win Strategy: More capacity, lower LR, force neighbor learning
# Target: 0.27-0.30 AUPRC (up from 0.24)

set -e

echo "=========================================="
echo "🚀 Aggressive Transformer k-NN Training"
echo "=========================================="
echo ""
echo "📊 Configuration Changes (Professional Optimization):"
echo "   • Attention heads: 6 with 2400-dim projection (balanced per-head capacity)"
echo "   • Transformer layers: 2 (same, avoid param explosion)"
echo "   • Feedforward dim: 1280 (was 1024) [+25%]"
echo "   • Max LR: 5e-4 (was 1e-3) [conservative for larger batch]"
echo "   • Batch size: 640 (was 512) [more positives per GPU]"
echo "   • Accumulate batches: 1 (was 2) [effective batch = 640×3 = 1,920]"
echo "   • Max epochs: 80 (unchanged) [peak typically ~epoch 25]"
echo "   • Patience: 15 (unchanged) [aggressive early stopping]"
echo "   • pct_start: 0.2 (was 0.3) [faster warmup for larger batch]"
echo "   • Modality dropout: 0.1 (was 0.0) [force neighbor use]"
echo "   • Residual gate: stronger α penalty (0.15→0.05), temp=1.5, dropout 25%→5%, target 0.55"
echo "   • Context scale anneals 0.5 → 1.0 over first 15 epochs"
echo "   • Neighbour-only aux head (0.10→0.02 over 12 epochs) keeps attention gradients alive"
echo "   • Distance clamp: 30Å (was 0) [better bias learning]"
echo "   • Validation logging: per-head attention entropy & normalized context norms"
echo ""
echo "🎯 Target: 0.27-0.29 AUPRC (scientific optimization)"
echo "📈 Expected: +3-5% over baseline, faster convergence, less overfit"
echo ""

# Kill any existing training
echo "🧹 Stopping existing training..."
tmux kill-session -t transformer_training 2>/dev/null || true
sleep 2

# Activate environment
echo "🔧 Activating conda environment..."
source ~/.bashrc
conda activate p2rank_env

# Resolve project root (directory containing this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check GPUs
echo ""
echo "📊 GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | \
    awk -F, '{printf "GPU %s: %s (%.0f GB free / %.0f GB total)\n", $1, $2, $4/1024, $3/1024}'

# Set GPUs (Volta architecture: 1,3,4)
export CUDA_VISIBLE_DEVICES=1,3,4

echo ""
echo "🚀 Launching aggressive transformer training..."
echo "   Experiment: fusion_transformer_aggressive"
echo "   GPUs: 1,3,4 (Volta)"
echo "   Log prefix: training_aggressive_(timestamp).log"
echo ""

# Launch in tmux
tmux new-session -d -s transformer_training bash -c "
cd \"$ROOT_DIR\" && \
export PROJECT_ROOT=\"$ROOT_DIR\" && \
export CUDA_VISIBLE_DEVICES=1,3,4 && \
python src/train.py experiment=fusion_transformer_aggressive 2>&1 | \
    tee training_aggressive_\$(date +%F_%H-%M-%S).log
"

echo "✅ Training launched in tmux session 'transformer_training'"
echo ""
echo "📊 Monitor with:"
echo "   tmux attach -t transformer_training"
echo "   tail -f training_aggressive_*.log"
echo ""
echo "🔍 W&B Dashboard:"
echo "   https://wandb.ai/max-hageneder-johannes-kepler-universit-t-linz/fusion_pocknet_thesis"
echo ""
echo "⏱️ Expected training time: ~18-24 hours (≤80 epochs)"
echo "🎯 Look for AUPRC > 0.27 by epoch 12-20"

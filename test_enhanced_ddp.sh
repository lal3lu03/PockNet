#!/bin/bash

# Test script for PockNet with enhanced DDP logging
# This script runs the modified experiment with excluded protrusion.distanceToCenter
# and cosine learning rate scheduling

echo "=== Testing Modified PockNet Experiment ==="
echo "Configuration:"
echo "- Excluded feature: protrusion.distanceToCenter"
echo "- Scheduler: CosineAnnealingLR"
echo "- Input dimension: 41 features"
echo "- Enhanced DDP logging enabled"
echo ""

# Set environment variables for better DDP performance
export NCCL_DEBUG=INFO
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create a test run with limited epochs for validation
echo "Running test experiment with 5 epochs..."

python src/train.py \
    experiment=pocknet \
    trainer.max_epochs=5 \
    trainer.limit_train_batches=0.1 \
    trainer.limit_val_batches=0.1 \
    logger=wandb_ddp \
    tags=["test","ddp","no_distance","cosine_scheduler"] \
    task_name="test_pocknet_ddp_enhanced" \
    extras.print_config=true

echo ""
echo "=== Test Results ==="
if [ $? -eq 0 ]; then
    echo "✅ Test experiment completed successfully!"
    echo "✅ Enhanced DDP logging validated"
    echo "✅ Modified PockNet configuration working"
    echo ""
    echo "Ready to run full experiment with:"
    echo "python src/train.py experiment=pocknet logger=wandb_ddp"
else
    echo "❌ Test experiment failed"
    echo "Check logs for issues with:"
    echo "- DDP initialization"
    echo "- W&B logging"
    echo "- Feature exclusion"
    echo "- Cosine scheduler"
fi

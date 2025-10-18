#!/bin/bash
# Run optimized H5 generation with all available CPU cores

# Activate conda environment
source ~/.bashrc
conda activate p2rank_env

# Get number of CPU cores
NCORES=$(nproc)

echo "=============================================="
echo "H5 Generation - OPTIMIZED Multi-Processing"
echo "=============================================="
echo "Using $NCORES CPU cores"
echo "Start time: $(date)"
echo ""

# Create output directory
mkdir -p data/h5

# Run with all cores and log output
python generate_h5_v2_optimized.py \
    --csv data/vectorsTrain_all_chainfix.csv \
    --esm_dir data/esm2_3B_chain \
    --pdb_base_dir data/p2rank-datasets \
    --bu48_txt data/bu48_proteins.txt \
    --out data/h5/all_train_transformer_v2_optimized.h5 \
    --k 3 \
    --workers $NCORES \
    --ds_file data/all_train.ds 2>&1 | tee h5_generation_optimized.log

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "Log saved to: h5_generation_optimized.log"
echo "Output file: data/h5/all_train_transformer_v2_optimized.h5"
echo "=============================================="

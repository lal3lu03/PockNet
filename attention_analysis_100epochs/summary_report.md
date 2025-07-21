
# PockNet Attention Diagnosis Report
Generated: 2025-06-03 15:11:03

## Training Summary
- Total epochs: 10
- Final train loss: 0.0750
- Final val loss: 0.0737
- Attention monitoring points: 10

## Attention Pattern Analysis

### Sparsity Evolution
- Initial sparsity: 0.205
- Final sparsity: 0.254
- Change: +0.049

### Key Findings
- Most important decision step: 0
- Entmax15 effectiveness: 0.0% zero attention
- Top 5 features: [34, 19, 5, 4, 7]

### Recommendations
- 💡 Consider increasing entmax temperature for better sparsity

### Files Generated
- Attention heatmaps: attention_analysis_100epochs/
- Evolution plots: attention_analysis_100epochs/attention_evolution.png
- Analysis data: attention_analysis_100epochs/attention_analysis.pkl

## Next Steps
1. Analyze top features for domain relevance
2. Consider feature engineering based on attention patterns
3. Experiment with different attention mechanisms if needed
4. Use insights for model architecture optimization

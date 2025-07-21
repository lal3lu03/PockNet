"""PockNetESM: ESM-augmented PockNet architecture for binding site prediction.

This module extends the PockNet architecture to incorporate ESM-2 protein embeddings
along with traditional tabular features for improved binding site prediction.
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pocknet import FeatureTransformer, AttentiveTransformer, GLU, TabNetBlock


class ESMProjector(nn.Module):
    """Projects ESM embeddings to a lower dimensional space."""
    
    def __init__(self, esm_dim: int, projection_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(esm_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
        )
        
    def forward(self, esm_features: torch.Tensor) -> torch.Tensor:
        """Project ESM features to lower dimensional space."""
        return self.projection(esm_features)


class AttentionFusion(nn.Module):
    """Attention-based fusion of tabular and ESM features."""
    
    def __init__(self, tabular_dim: int, esm_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.tabular_attention = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.esm_attention = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Output projection to common dimension
        self.output_dim = max(tabular_dim, esm_dim)
        self.tabular_proj = nn.Linear(tabular_dim, self.output_dim) if tabular_dim != self.output_dim else nn.Identity()
        self.esm_proj = nn.Linear(esm_dim, self.output_dim) if esm_dim != self.output_dim else nn.Identity()
        
    def forward(self, tabular_features: torch.Tensor, esm_features: torch.Tensor) -> torch.Tensor:
        """Fuse tabular and ESM features using attention."""
        # Compute attention weights
        tabular_weight = self.tabular_attention(tabular_features)
        esm_weight = self.esm_attention(esm_features)
        
        # Normalize weights
        total_weight = tabular_weight + esm_weight + 1e-8
        tabular_weight = tabular_weight / total_weight
        esm_weight = esm_weight / total_weight
        
        # Project to common dimension and fuse
        tabular_proj = self.tabular_proj(tabular_features)
        esm_proj = self.esm_proj(esm_features)
        
        fused = tabular_weight * tabular_proj + esm_weight * esm_proj
        return fused


class GatedFusion(nn.Module):
    """Gated fusion of tabular and ESM features."""
    
    def __init__(self, tabular_dim: int, esm_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.output_dim = max(tabular_dim, esm_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(tabular_dim + esm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
            nn.Sigmoid()
        )
        
        # Feature projections
        self.tabular_proj = nn.Linear(tabular_dim, self.output_dim) if tabular_dim != self.output_dim else nn.Identity()
        self.esm_proj = nn.Linear(esm_dim, self.output_dim) if esm_dim != self.output_dim else nn.Identity()
        
    def forward(self, tabular_features: torch.Tensor, esm_features: torch.Tensor) -> torch.Tensor:
        """Fuse tabular and ESM features using gating."""
        # Compute gate values
        combined_input = torch.cat([tabular_features, esm_features], dim=1)
        gate_values = self.gate(combined_input)
        
        # Project features to common dimension
        tabular_proj = self.tabular_proj(tabular_features)
        esm_proj = self.esm_proj(esm_features)
        
        # Apply gating
        fused = gate_values * tabular_proj + (1 - gate_values) * esm_proj
        return fused


class PockNetESM(nn.Module):
    """ESM-augmented PockNet architecture for protein binding site prediction.
    
    This model combines traditional tabular features with ESM-2 protein embeddings
    using various fusion strategies within the TabNet-inspired architecture.
    
    Args:
        tabular_dim: Dimension of tabular features
        esm_dim: Dimension of ESM embeddings
        output_dim: Number of output classes (1 for binary classification)
        n_steps: Number of decision steps
        n_d: Dimension of decision features
        n_a: Dimension of attention features
        n_shared: Number of shared layers in feature transformer
        n_independent: Number of independent layers per step
        gamma: Coefficient for attention regularization
        epsilon: Small constant for numerical stability
        dropout: Dropout rate
        esm_projection_dim: Dimension to project ESM embeddings to
        fusion_strategy: How to combine tabular and ESM features
    """
    
    def __init__(
        self,
        tabular_dim: int,
        esm_dim: int,
        output_dim: int = 1,
        n_steps: int = 3,
        n_d: int = 8,
        n_a: int = 8,
        n_shared: int = 2,
        n_independent: int = 2,
        gamma: float = 1.3,
        epsilon: float = 1e-15,
        dropout: float = 0.0,
        esm_projection_dim: int = 128,
        fusion_strategy: str = "concatenate",
    ):
        super().__init__()
        
        self.tabular_dim = tabular_dim
        self.esm_dim = esm_dim
        self.output_dim = output_dim
        self.n_steps = n_steps
        self.n_d = n_d
        self.n_a = n_a
        self.gamma = gamma
        self.epsilon = epsilon
        self.fusion_strategy = fusion_strategy
        
        # ESM embedding projector
        self.esm_projector = ESMProjector(esm_dim, esm_projection_dim, dropout)
        
        # Feature fusion
        if fusion_strategy == "concatenate":
            self.input_dim = tabular_dim + esm_projection_dim
            self.fusion = None
        elif fusion_strategy == "attention":
            self.fusion = AttentionFusion(tabular_dim, esm_projection_dim)
            self.input_dim = self.fusion.output_dim
        elif fusion_strategy == "gated":
            self.fusion = GatedFusion(tabular_dim, esm_projection_dim)
            self.input_dim = self.fusion.output_dim
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # Input batch normalization
        self.initial_bn = nn.BatchNorm1d(self.input_dim)
        
        # TabNet blocks for each decision step
        self.tabnet_blocks = nn.ModuleList([
            TabNetBlock(
                input_dim=self.input_dim,
                output_dim=n_d + n_a,
                n_independent=n_independent,
                n_shared=n_shared,
                dropout=dropout
            ) for _ in range(n_steps)
        ])
        
        # Final classifier
        self.final_classifier = nn.Linear(n_d * n_steps, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, tabular_features: torch.Tensor, esm_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through ESM-augmented PockNet.
        
        Args:
            tabular_features: Tabular features [batch_size, tabular_dim]
            esm_features: ESM embeddings [batch_size, esm_dim]
            
        Returns:
            output: Final predictions [batch_size, output_dim]
            attention_weights: Attention weights for interpretability [batch_size, n_steps, input_dim]
        """
        batch_size = tabular_features.size(0)
        
        # Project ESM features
        esm_projected = self.esm_projector(esm_features)
        
        # Fuse features
        if self.fusion_strategy == "concatenate":
            x = torch.cat([tabular_features, esm_projected], dim=1)
        else:
            x = self.fusion(tabular_features, esm_projected)
        
        # Initial batch normalization
        x = self.initial_bn(x)
        
        # Initialize priors (uniform attention initially)
        priors = torch.ones_like(x)
        
        # Store outputs from each step
        step_outputs = []
        attention_weights = []
        
        # Process through each TabNet block
        for i, tabnet_block in enumerate(self.tabnet_blocks):
            step_output, masked_features, priors, attention_mask = tabnet_block(x, priors)
            
            # Split into decision and attention parts
            decision_out = step_output[:, :self.n_d]
            
            step_outputs.append(decision_out)
            attention_weights.append(attention_mask)
        
        # Aggregate all step outputs
        aggregated_output = torch.cat(step_outputs, dim=1)  # [batch_size, n_d * n_steps]
        
        # Final classification
        output = self.final_classifier(aggregated_output)
        
        # Stack attention weights
        attention_weights = torch.stack(attention_weights, dim=1)  # [batch_size, n_steps, input_dim]
        
        return output, attention_weights
    
    def forward_masks(self, tabular_features: torch.Tensor, esm_features: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only the output (for compatibility)."""
        output, _ = self.forward(tabular_features, esm_features)
        return output

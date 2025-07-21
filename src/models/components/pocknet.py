"""PockNet: Custom TabNet-inspired architecture for binding site prediction.

This is a from-scratch implementation of the TabNet architecture, adapted for protein binding site prediction.
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureTransformer(nn.Module):
    """Feature transformer block with shared and decision-specific features."""
    
    def __init__(self, input_dim: int, output_dim: int, shared_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        
        self.shared_layers = nn.ModuleList()
        self.decision_layers = nn.ModuleList()
        
        # Shared feature processing
        current_dim = input_dim
        for i in range(shared_layers):
            self.shared_layers.append(nn.Linear(current_dim, output_dim))
            self.shared_layers.append(nn.BatchNorm1d(output_dim))
            self.shared_layers.append(nn.ReLU())
            if dropout > 0:
                self.shared_layers.append(nn.Dropout(dropout))
            current_dim = output_dim
            
        # Decision-specific processing
        self.decision_layers.append(nn.Linear(current_dim, output_dim))
        self.decision_layers.append(nn.BatchNorm1d(output_dim))
        self.decision_layers.append(nn.ReLU())
        if dropout > 0:
            self.decision_layers.append(nn.Dropout(dropout))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shared processing
        shared_out = x
        for layer in self.shared_layers:
            shared_out = layer(shared_out)
        
        # Decision-specific processing
        decision_out = shared_out
        for layer in self.decision_layers:
            decision_out = layer(decision_out)
            
        return shared_out, decision_out

class AttentiveTransformer(nn.Module):
    """Attentive transformer for feature selection."""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0): # input_dim is (n_d+n_a), output_dim is original_input_dim
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.bn = nn.BatchNorm1d(output_dim) # BatchNorm should be on the output of the linear layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, priors: torch.Tensor, processed_feat: torch.Tensor) -> torch.Tensor:
        # processed_feat is the output of the shared part of FeatureTransformer (dim: n_d + n_a)
        # priors has dimensions of the original input features (dim: original_input_dim)
        # self.fc maps from (n_d + n_a) to original_input_dim
        
        attention_logits = self.fc(processed_feat) # Shape: [batch_size, original_input_dim]
        attention_logits = self.dropout(attention_logits) # Add dropout for regularization
        normalized_logits = self.bn(attention_logits) # Apply BatchNorm. Shape: [batch_size, original_input_dim]
        
        # Multiply by priors before applying sigmoid
        # priors act as a scale for the attention logits for each feature
        final_logits = priors * normalized_logits # Shape: [batch_size, original_input_dim]
        return torch.sigmoid(final_logits)  # Changed from entmax15 to sigmoid for better generalization


class GLU(nn.Module):
    """Gated Linear Unit activation."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        return out[:, :out.size(1)//2] * torch.sigmoid(out[:, out.size(1)//2:])


class TabNetBlock(nn.Module):
    """Single TabNet decision step block."""
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, # This is n_d + n_a
        n_independent: int = 2, 
        n_shared: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.feature_transformer = FeatureTransformer(
            input_dim, output_dim, shared_layers=n_shared, dropout=dropout # output_dim here is n_d + n_a
        )
        
        # AttentiveTransformer input_dim should be the output_dim of FeatureTransformer's shared part (which is n_d + n_a)
        # AttentiveTransformer output_dim should be the original input_dim to create a mask of the same size as features
        self.attentive_transformer = AttentiveTransformer(output_dim, input_dim, dropout)
        
        # GLU blocks for decision making
        self.glu = GLU(output_dim, output_dim)
        
        # Independent layers for this step
        self.independent_layers = nn.ModuleList()
        current_dim = output_dim
        for i in range(n_independent):
            self.independent_layers.append(nn.Linear(current_dim, output_dim))
            self.independent_layers.append(nn.BatchNorm1d(output_dim))
            self.independent_layers.append(nn.ReLU())
            if dropout > 0:
                self.independent_layers.append(nn.Dropout(dropout))
            current_dim = output_dim
    
    def forward(self, features: torch.Tensor, priors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: # Added torch.Tensor for mask
        # Transform features
        shared_out, decision_out = self.feature_transformer(features)
        
        # Apply attention
        mask = self.attentive_transformer(priors, shared_out)
        masked_features = features * mask
        
        # Apply GLU
        glu_out = self.glu(decision_out)
        
        # Independent processing for this step
        step_output = glu_out
        for layer in self.independent_layers:
            step_output = layer(step_output)
        
        # Update priors for next step
        # Encourage sparsity by reducing attention on used features
        new_priors = priors * (1.0 - mask + 1e-10)  # Small epsilon to avoid numerical issues
        
        return step_output, masked_features, new_priors, mask # Return mask


class PockNet(nn.Module):
    """PockNet: TabNet-inspired architecture for protein binding site prediction.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output classes (1 for binary classification)
        n_steps: Number of decision steps
        n_d: Dimension of decision features
        n_a: Dimension of attention features  
        n_shared: Number of shared layers in feature transformer
        n_independent: Number of independent layers per step
        gamma: Coefficient for attention regularization
        epsilon: Small constant for numerical stability
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        n_steps: int = 3,
        n_d: int = 8,
        n_a: int = 8,
        n_shared: int = 2,
        n_independent: int = 2,
        gamma: float = 1.3,
        epsilon: float = 1e-15,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_steps = n_steps
        self.n_d = n_d
        self.n_a = n_a
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Input batch normalization
        self.initial_bn = nn.BatchNorm1d(input_dim)
        
        # TabNet blocks for each decision step
        self.tabnet_blocks = nn.ModuleList([
            TabNetBlock(
                input_dim=input_dim,
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through PockNet.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            output: Final predictions [batch_size, output_dim]
            attention_weights: Attention weights for interpretability [batch_size, n_steps, input_dim]
        """
        batch_size = x.size(0)
        
        # Initial batch normalization
        x = self.initial_bn(x)
        
        # Initialize priors (uniform attention initially)
        priors = torch.ones_like(x)
        
        # Store outputs from each step
        step_outputs = []
        attention_weights = []
        
        # Process through each TabNet block
        for i, tabnet_block in enumerate(self.tabnet_blocks):
            step_output, masked_features, priors, attention_mask_from_block = tabnet_block(x, priors) # Capture actual mask
            
            # Split into decision and attention parts
            decision_out = step_output[:, :self.n_d]
            # attention_out = step_output[:, self.n_d:] # This line is no longer needed for attention_weights
            
            step_outputs.append(decision_out)
            
            # Store attention weights for interpretability
            # attention_mask = torch.softmax(attention_out @ torch.ones(self.n_a, self.input_dim).to(x.device), dim=-1) # Old way
            attention_weights.append(attention_mask_from_block) # Store the actual mask from AttentiveTransformer
        
        # Aggregate all step outputs
        aggregated_output = torch.cat(step_outputs, dim=1)  # [batch_size, n_d * n_steps]
        
        # Final classification
        output = self.final_classifier(aggregated_output)
        
        # Stack attention weights
        attention_weights = torch.stack(attention_weights, dim=1)  # [batch_size, n_steps, input_dim]
        
        return output, attention_weights
    
    def forward_masks(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only the output (for compatibility)."""
        output, _ = self.forward(x)
        return output

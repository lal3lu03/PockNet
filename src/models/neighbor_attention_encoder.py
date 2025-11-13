"""
Neighbor Attention Encoder for k-NN ESM Embedding Aggregation.

This module implements transformer-based aggregation of k nearest neighbor
ESM embeddings with distance-aware attention biasing. The encoder learns
to selectively attend to relevant neighbors based on both embedding similarity
and spatial distance.

Architecture:
    Input:  [B, k, D] neighbor embeddings + [B, k] distances
    Process: Multi-head attention with distance bias + FFN
    Output: [B, D] aggregated embedding

Key Features:
    - Distance-aware attention biasing (learned, exponential, or none)
    - Multi-head self-attention for diverse neighbor interactions
    - Feed-forward network for non-linear feature transformation
    - Layer normalization and residual connections
    - Handles variable k with padding/masking

Usage:
    encoder = NeighborAttentionEncoder(d_model=2560, num_heads=8, num_layers=2)
    neighbors = torch.randn(B, k, 2560)  # k neighbor embeddings
    distances = torch.rand(B, k) * 10    # distances in Angstroms
    aggregated = encoder(neighbors, distances=distances)  # [B, 2560]

Author: AI Scientific Assistant
Date: January 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal, Dict, List, Union, Tuple
import logging

log = logging.getLogger(__name__)


class DistanceBias(nn.Module):
    """
    Distance-based bias for attention scores.
    
    Converts spatial distances (Angstroms) into attention biases that modulate
    the attention scores based on spatial proximity.
    """
    
    def __init__(
        self,
        bias_type: Literal["learned", "exponential", "none"] = "learned",
        num_heads: int = 8,
        sigma: float = 5.0
    ):
        """
        Args:
            bias_type: Type of distance bias
                - "learned": Learnable MLP transforms distances to bias
                - "exponential": Fixed exponential decay exp(-d²/2σ²)
                - "none": No distance bias (standard attention)
            num_heads: Number of attention heads
            sigma: Standard deviation for exponential decay (Angstroms)
        """
        super().__init__()
        self.bias_type = bias_type
        self.num_heads = num_heads
        self.sigma = sigma
        
        if bias_type == "learned":
            # MLP: distances [B,k] → bias [B,num_heads,k,k]
            self.mlp = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, num_heads)
            )
        elif bias_type == "exponential":
            # Learnable sigma parameter
            self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)))
        elif bias_type == "none":
            pass
        else:
            raise ValueError(f"Unknown bias_type: {bias_type}")
    
    def forward(self, distances: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Compute distance bias for attention.
        
        Args:
            distances: [B, k] distances in Angstroms
            
        Returns:
            bias: [B, num_heads, k, k] attention bias (or None for bias_type="none")
        """
        if self.bias_type == "none":
            return None
        
        B, k = distances.shape
        
        if self.bias_type == "learned":
            # Expand distances for MLP: [B, k, 1]
            d_expanded = distances.unsqueeze(-1)
            
            # Apply MLP: [B, k, num_heads]
            bias_per_head = self.mlp(d_expanded)
            
            # Reshape to [B, num_heads, k, 1] and broadcast to [B, num_heads, k, k]
            # Interpretation: bias_ij = f(distance_j) affects attention from i to j
            bias = bias_per_head.transpose(1, 2).unsqueeze(-1)  # [B, num_heads, k, 1]
            bias = bias.expand(B, self.num_heads, k, k)
            
            return bias
        
        elif self.bias_type == "exponential":
            # Exponential decay: exp(-d²/2σ²)
            sigma = torch.exp(self.log_sigma)
            
            # Compute pairwise distance bias: [B, k, k]
            # For simplicity, use distances to compute symmetric bias
            dist_matrix = distances.unsqueeze(1) + distances.unsqueeze(2)  # [B, k, k]
            bias = torch.exp(-dist_matrix**2 / (2 * sigma**2))
            
            # Add head dimension: [B, 1, k, k] → [B, num_heads, k, k]
            bias = bias.unsqueeze(1).expand(B, self.num_heads, k, k)
            
            # Convert to log space for adding to attention logits
            bias = torch.log(bias + 1e-8)
            
            return bias


class NeighborAttentionLayer(nn.Module):
    """
    Single transformer layer for neighbor attention.
    
    Consists of:
    1. Multi-head self-attention with distance bias
    2. Feed-forward network
    3. Layer normalization and residual connections
    """
    
    def __init__(
        self,
        d_model: int = 2560,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        distance_bias: Optional[DistanceBias] = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        # Multi-head attention components
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Distance bias module
        self.distance_bias = distance_bias
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        distances: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        collect_stats: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through attention layer.
        
        Args:
            x: [B, k, d_model] neighbor embeddings
            distances: [B, k] distances for attention bias
            mask: [B, k] boolean mask (True = valid, False = padding)
            
        Returns:
            out: [B, k, d_model] attended embeddings
        """
        B, k, d = x.shape
        
        # Self-attention with residual
        attn_result = self._self_attention(x, distances, mask, collect_stats=collect_stats)
        if collect_stats:
            attn_out, stats = attn_result
        else:
            attn_out = attn_result
            stats = None
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return (x, stats) if collect_stats else x
    
    def _self_attention(
        self,
        x: torch.Tensor,
        distances: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        collect_stats: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Compute multi-head self-attention."""
        B, k, d = x.shape
        
        # Project to Q, K, V: [B, k, 3*d] → 3 × [B, k, d]
        qkv = self.qkv_proj(x)
        q, k_proj, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head: [B, k, d] → [B, num_heads, k, head_dim]
        q = q.view(B, k, self.num_heads, self.head_dim).transpose(1, 2)
        k_proj = k_proj.view(B, k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention: QK^T / sqrt(d_k)
        scores = torch.matmul(q, k_proj.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # [B, num_heads, k, k]
        
        # Add distance bias if available
        if distances is not None and self.distance_bias is not None:
            distance_bias = self.distance_bias(distances)
            if distance_bias is not None:
                scores = scores + distance_bias
        
        # Apply mask if provided (mask out padding)
        if mask is not None:
            # mask: [B, k] → [B, 1, 1, k] for broadcasting
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, k]
            # Use -1e4 instead of -1e9 for float16 compatibility (max ~65k)
            scores = scores.masked_fill(~mask_expanded, -1e4)
        
        # Softmax attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v)  # [B, num_heads, k, head_dim]
        
        # Reshape back: [B, num_heads, k, head_dim] → [B, k, d]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, k, d)
        
        # Output projection
        attn_out = self.out_proj(attn_out)
        
        if not collect_stats:
            return attn_out
        
        probs = attn_weights.clamp(min=1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1)  # [B, num_heads, k]
        if mask is not None:
            qmask = mask.unsqueeze(1)  # [B, 1, k]
            entropy = entropy * qmask
            denom = qmask.sum(dim=(0, 2)).clamp(min=1.0)
            head_entropy = entropy.sum(dim=(0, 2)) / denom
        else:
            head_entropy = entropy.mean(dim=(0, 2))
        
        stats = {"entropy_per_head": head_entropy}
        return attn_out, stats


class NeighborAttentionEncoder(nn.Module):
    """
    Transformer encoder for aggregating k-NN ESM embeddings.
    
    Takes k neighbor embeddings and their distances, applies multi-layer
    attention with distance biasing, and aggregates to a single embedding.
    
    Architecture:
        Input [B, k, 2560] → Attention Layers → Pooling → Output [B, 2560]
    """
    
    def __init__(
        self,
        d_model: int = 2560,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        distance_bias_type: Literal["learned", "exponential", "none"] = "learned",
        pooling: Literal["mean", "max", "cls"] = "mean",
        sigma: float = 5.0,
        attention_dim: Optional[int] = None
    ):
        """
        Args:
            d_model: Embedding dimension (ESM2-3B uses 2560)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Hidden dimension in FFN
            dropout: Dropout probability
            distance_bias_type: Type of distance bias ("learned", "exponential", "none")
            pooling: Aggregation method ("mean", "max", "cls")
            sigma: Initial sigma for exponential distance bias (Angstroms)
            attention_dim: Optional reduced dimension for attention. If provided and different
                from d_model, inputs are projected to this dimension for attention and then
                projected back to d_model on output.
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pooling = pooling
        self.attention_dim = attention_dim or d_model
        if self.attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim ({self.attention_dim}) must be divisible by num_heads ({num_heads})"
            )
        
        self.input_proj = None
        self.output_proj = None
        if self.attention_dim != d_model:
            self.input_proj = nn.Linear(d_model, self.attention_dim)
            self.output_proj = nn.Linear(self.attention_dim, d_model)
        
        # Distance bias module (shared across layers)
        if distance_bias_type != "none":
            self.distance_bias = DistanceBias(
                bias_type=distance_bias_type,
                num_heads=num_heads,
                sigma=sigma
            )
        else:
            self.distance_bias = None
        
        # Stack of attention layers
        self.layers = nn.ModuleList([
            NeighborAttentionLayer(
                d_model=self.attention_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                distance_bias=self.distance_bias
            )
            for _ in range(num_layers)
        ])
        
        # Optional CLS token for pooling
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.attention_dim))
        
        # Final layer norm
        self.norm = nn.LayerNorm(self.attention_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        neighbors: torch.Tensor,
        distances: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Aggregate k neighbor embeddings into single embedding.
        
        Args:
            neighbors: [B, k, d_model] k neighbor ESM embeddings
            distances: [B, k] distances to neighbors (Angstroms)
            mask: [B, k] boolean mask (True = valid, False = padding)
            
        Returns:
            aggregated: [B, d_model] aggregated embedding
        """
        B, k, d = neighbors.shape
        
        # Validate input dimensions
        assert d == self.d_model, f"Expected d_model={self.d_model}, got {d}"
        
        # Create default mask if not provided
        if mask is None:
            mask = torch.ones(B, k, dtype=torch.bool, device=neighbors.device)
        
        # Apply optional projection to attention dimension
        if self.input_proj is not None:
            x = self.input_proj(neighbors)
        else:
            x = neighbors
        
        # Add CLS token if using CLS pooling (after projection so dimensions match)
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, attention_dim]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, k+1, attention_dim]
            
            # Extend mask for CLS token (always valid)
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)  # [B, k+1]
            
            # Extend distances for CLS token (zero distance)
            if distances is not None:
                cls_dist = torch.zeros(B, 1, device=distances.device)
                distances = torch.cat([cls_dist, distances], dim=1)  # [B, k+1]
        
        entropy_layers: List[torch.Tensor] = []
        for layer in self.layers:
            if return_attention:
                x, layer_stats = layer(x, distances=distances, mask=mask, collect_stats=True)
                if layer_stats and "entropy_per_head" in layer_stats:
                    entropy_layers.append(layer_stats["entropy_per_head"])
            else:
                x = layer(x, distances=distances, mask=mask)
        
        # Final normalization
        x = self.norm(x)
        
        # Aggregate to single embedding
        if self.pooling == "mean":
            # Mean pooling over valid neighbors
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()  # [B, k, 1]
                x_masked = x * mask_expanded
                aggregated = x_masked.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
            else:
                aggregated = x.mean(dim=1)
        
        elif self.pooling == "max":
            # Max pooling over valid neighbors
            if mask is not None:
                x_masked = x.masked_fill(~mask.unsqueeze(-1), -1e9)
                aggregated = x_masked.max(dim=1)[0]
            else:
                aggregated = x.max(dim=1)[0]
        
        elif self.pooling == "cls":
            # Use CLS token as aggregated embedding
            aggregated = x[:, 0, :]  # [B, d_model]
        
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        if self.output_proj is not None:
            aggregated = self.output_proj(aggregated)
        
        if not return_attention:
            return aggregated  # [B, d_model]
        
        stats: Dict[str, torch.Tensor] = {}
        if entropy_layers:
            entropy_stack = torch.stack(entropy_layers)  # [num_layers, num_heads]
            stats["entropy_per_layer"] = entropy_stack
            stats["entropy_per_head"] = entropy_stack.mean(dim=0)
        else:
            stats["entropy_per_head"] = torch.zeros(self.num_heads, device=aggregated.device)
        
        return aggregated, stats


def create_neighbor_attention_encoder(
    d_model: int = 2560,
    num_heads: int = 8,
    num_layers: int = 2,
    **kwargs
) -> NeighborAttentionEncoder:
    """
    Factory function for creating NeighborAttentionEncoder with common defaults.
    
    Args:
        d_model: Embedding dimension (default: 2560 for ESM2-3B)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 2)
        **kwargs: Additional arguments passed to NeighborAttentionEncoder
        
    Returns:
        encoder: NeighborAttentionEncoder instance
    """
    return NeighborAttentionEncoder(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    # Example: Create encoder and test forward pass
    B, k, d = 16, 3, 2560  # Batch=16, k=3 neighbors, d=2560 (ESM2-3B)
    
    # Create encoder
    encoder = NeighborAttentionEncoder(
        d_model=d,
        num_heads=8,
        num_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        distance_bias_type="learned",
        pooling="mean"
    )
    
    # Create dummy inputs
    neighbors = torch.randn(B, k, d)  # Random neighbor embeddings
    distances = torch.rand(B, k) * 10  # Random distances (0-10 Angstroms)
    
    # Forward pass
    aggregated = encoder(neighbors, distances=distances)
    
    print(f"Input shape: {neighbors.shape}")
    print(f"Distances shape: {distances.shape}")
    print(f"Output shape: {aggregated.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Verify output properties
    assert aggregated.shape == (B, d), f"Expected shape ({B}, {d}), got {aggregated.shape}"
    assert not torch.isnan(aggregated).any(), "Output contains NaN values"
    assert not torch.isinf(aggregated).any(), "Output contains Inf values"
    
    print("NeighborAttentionEncoder test passed!")

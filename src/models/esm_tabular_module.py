"""
ESM + Tabular Feature Fusion Model for Binding Site Prediction.

This model combines:
1. Tabular features (39D) from physicochemical properties  
2. ESM2 embeddings (2560D) with proper residue-specific mapping
3. Multiple fusion strategies optimized for imbalanced binary classification
4. BCE with pos_weight for severe class imbalance (2.5% positive rate)
5. Dynamic threshold optimization on validation set
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from lightning import LightningModule
from torchmetrics import (
    Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision, 
    JaccardIndex, ConfusionMatrix, Specificity
)
from torchmetrics.functional import auroc as tm_auroc, average_precision as tm_average_precision
from torch.nn import BCEWithLogitsLoss
from typing import Dict, Any, Optional, List, Literal
import logging

log = logging.getLogger(__name__)

# Import k-NN utilities
try:
    from src.data.knn_utils import aggregate_knn_embeddings
except ImportError:
    log.warning("k-NN utilities not available, using single nearest residue")

# Import NeighborAttentionEncoder for transformer aggregation
try:
    from src.models.neighbor_attention_encoder import NeighborAttentionEncoder
except ImportError:
    log.warning("NeighborAttentionEncoder not available, transformer aggregation disabled")
    NeighborAttentionEncoder = None


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance by down-weighting easy examples."""
    
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        w = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = -w * (1 - pt).pow(self.gamma) * torch.log(pt.clamp(min=self.eps))
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multilabel problems, adapted for binary classification."""
    
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
    
    def forward(self, logits, targets):
        x = torch.sigmoid(logits)
        xs_pos = x
        xs_neg = 1 - x
        
        if self.clip and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps)) * ((1 - xs_pos) ** self.gamma_pos)
        loss_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps)) * (x ** self.gamma_neg)
        loss = -(loss_pos + loss_neg)
        return loss.mean()


class TabularEncoder(nn.Module):
    """Encoder for tabular features."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x):
        return self.encoder(x)


class EsmEncoder(nn.Module):
    """Encoder for ESM embeddings."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x):
        return self.encoder(x)


class AttentionFusion(nn.Module):
    """Attention-based fusion of tabular and ESM features."""
    
    def __init__(self, tabular_dim: int, esm_dim: int, fusion_dim: int):
        super().__init__()
        
        self.tabular_proj = nn.Linear(tabular_dim, fusion_dim)
        self.esm_proj = nn.Linear(esm_dim, fusion_dim)
        self.attention = nn.MultiheadAttention(fusion_dim, num_heads=8, batch_first=True)
        self.output_dim = fusion_dim
    
    def forward(self, tabular_feat, esm_feat):
        # Project to fusion dimension
        tab_proj = self.tabular_proj(tabular_feat).unsqueeze(1)  # [B, 1, fusion_dim]
        esm_proj = self.esm_proj(esm_feat).unsqueeze(1)  # [B, 1, fusion_dim]
        
        # Concatenate for attention
        combined = torch.cat([tab_proj, esm_proj], dim=1)  # [B, 2, fusion_dim]
        
        # Apply attention
        attended, _ = self.attention(combined, combined, combined)
        
        # Pool attended features
        fused = attended.mean(dim=1)  # [B, fusion_dim]
        
        return fused


class GatedFusion(nn.Module):
    """Gated fusion of tabular and ESM features."""
    
    def __init__(self, tabular_dim: int, esm_dim: int, fusion_dim: int):
        super().__init__()
        
        self.tabular_proj = nn.Linear(tabular_dim, fusion_dim)
        self.esm_proj = nn.Linear(esm_dim, fusion_dim)
        self.gate = nn.Linear(tabular_dim + esm_dim, fusion_dim)
        self.output_dim = fusion_dim
    
    def forward(self, tabular_feat, esm_feat):
        # Project features
        tab_proj = self.tabular_proj(tabular_feat)
        esm_proj = self.esm_proj(esm_feat)
        
        # Compute gate
        gate_input = torch.cat([tabular_feat, esm_feat], dim=-1)
        gate_weights = torch.sigmoid(self.gate(gate_input))
        
        # Apply gating
        fused = gate_weights * tab_proj + (1 - gate_weights) * esm_proj
        
        return fused


class LateFusionHead(nn.Module):
    """Late fusion head that blends per-modality logits."""
    
    def __init__(self, in_tab: int, in_esm: int):
        super().__init__()
        self.head_tab = nn.Linear(in_tab, 1)
        self.head_esm = nn.Linear(in_esm, 1)
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))  # α = sigmoid(alpha_raw)

    def forward(self, h_tab, h_esm):
        z_t = self.head_tab(h_tab)
        z_e = self.head_esm(h_esm)
        alpha = torch.sigmoid(self.alpha_raw)
        return alpha * z_e + (1 - alpha) * z_t  # [B, 1]


class FiLMFusion(nn.Module):
    """Feature-wise linear modulation of ESM by tabular."""
    
    def __init__(self, tabular_dim: int, esm_dim: int, fusion_dim: int):
        super().__init__()
        self.tabular_proj = nn.Linear(tabular_dim, fusion_dim)
        self.esm_proj = nn.Linear(esm_dim, fusion_dim)
        self.gamma = nn.Linear(tabular_dim, fusion_dim)
        self.beta = nn.Linear(tabular_dim, fusion_dim)
        self.output_dim = fusion_dim
    
    def forward(self, tabular_feat, esm_feat):
        t = self.tabular_proj(tabular_feat)
        e = self.esm_proj(esm_feat)
        e = self.gamma(tabular_feat).sigmoid() * e + self.beta(tabular_feat)  # FiLM
        return F.relu(e + t)  # residual combine


class EsmTabularModule(LightningModule):
    """Lightning module for ESM + Tabular feature fusion with BCE for imbalanced data."""
    
    def __init__(
        self,
        tabular_dim: int = 39,
        esm_dim: int = 2560,
        hidden_dims: List[int] = [1024, 512, 256],
        dropout: float = 0.2,
        fusion_method: str = "concat",
        fusion_dim: int = 512,
        num_classes: int = 1,  # Single logit for BCE
        loss_type: str = "bce",
        initial_pos_prior: float = 0.0253,
        modality_dropout_p: float = 0.0,  # Optional modality dropout for robustness
        # ⭐ NEW: Aggregation mode selection
        aggregation_mode: Literal["mean", "transformer"] = "mean",
        # k-NN ESM aggregation parameters (for mean mode)
        k_res_neighbors: int = 1,          # Number of nearest residues to aggregate
        neighbor_weighting: str = "softmax",  # "softmax", "inverse", "uniform"
        neighbor_temp: float = 2.0,        # Temperature for softmax weighting (Å)
        # ⭐ NEW: Transformer aggregation parameters
        neighbor_attention_heads: int = 8,
        neighbor_attention_layers: int = 2,
        neighbor_attention_dim_ff: int = 1024,
        neighbor_attention_dropout: float = 0.1,
        distance_bias_type: Literal["learned", "exponential", "none"] = "learned",
        attention_pooling: Literal["mean", "max", "cls"] = "mean",
        neighbor_attention_proj_dim: Optional[int] = None,
        # Gating regularisation & stabilisation
        gate_penalty_weight: float = 0.02,
        gate_penalty_target: float = 0.6,
        gate_temperature: float = 1.0,
        gate_dropout_p: float = 0.0,
        gate_dropout_target: float = 0.6,
        neighbor_context_scale: float = 1.0,
        gate_penalty_start: Optional[float] = None,
        gate_penalty_end: Optional[float] = None,
        gate_dropout_start: Optional[float] = None,
        gate_dropout_end: Optional[float] = None,
        neighbor_context_scale_start: Optional[float] = None,
        neighbor_context_scale_end: Optional[float] = None,
        gate_schedule_epochs: int = 0,
        aux_neighbor_head_weight: float = 0.0,
        aux_neighbor_head_weight_start: Optional[float] = None,
        aux_neighbor_head_weight_end: Optional[float] = None,
        aux_neighbor_head_schedule_epochs: int = 0,
        aux_distance_temperature: float = 5.0,
        optimizer: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # ⭐ NEW: Instantiate NeighborAttentionEncoder if transformer mode
        if aggregation_mode == "transformer":
            if NeighborAttentionEncoder is None:
                raise RuntimeError(
                    "Transformer aggregation mode requested but NeighborAttentionEncoder not available. "
                    "Ensure src/models/neighbor_attention_encoder.py is present."
                )
            self.neighbor_encoder = NeighborAttentionEncoder(
                d_model=esm_dim,
                num_heads=neighbor_attention_heads,
                num_layers=neighbor_attention_layers,
                dim_feedforward=neighbor_attention_dim_ff,
                dropout=neighbor_attention_dropout,
                distance_bias_type=distance_bias_type,
                pooling=attention_pooling,
                attention_dim=neighbor_attention_proj_dim
            )
            # Learned gating network (per-sample) to blend centre & neighbor context
            gate_hidden = 128
            self.center_norm = nn.LayerNorm(esm_dim)
            self.neighbor_norm = nn.LayerNorm(esm_dim)
            self.context_blend_norm = nn.LayerNorm(esm_dim)
            self.center_dropout = nn.Dropout(0.1)
            self.gate_mlp = nn.Sequential(
                nn.Linear(esm_dim * 2, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, 1)
            )
            # Initialise final gate layer to zero → sigmoid ≈ 0.5
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.zeros_(self.gate_mlp[-1].bias)
            log.info(f"✅ Initialized NeighborAttentionEncoder: {neighbor_attention_heads} heads, "
                     f"{neighbor_attention_layers} layers, distance_bias={distance_bias_type}")
            log.info("✅ Initialized sample-wise gating network for neighbour aggregation")
        else:
            self.neighbor_encoder = None
            self.center_norm = None
            self.neighbor_norm = None
            self.context_blend_norm = None
            self.center_dropout = None
            self.gate_mlp = None
        
        # Encoders
        self.tabular_encoder = TabularEncoder(tabular_dim, hidden_dims, dropout)
        self.center_encoder = EsmEncoder(esm_dim, hidden_dims, dropout)
        if aggregation_mode == "transformer":
            self.context_encoder = EsmEncoder(esm_dim, hidden_dims, dropout)
        else:
            self.context_encoder = None
        aux_in_dim = self.context_encoder.output_dim if self.context_encoder is not None else hidden_dims[-1]
        self._last_alpha_mean = None
        self._last_gate_dropout = None
        self._last_aux_loss = None
        self._last_attn_entropy_mean = None
        self._cached_neighbor_metrics: Dict[str, torch.Tensor] = {}
        self._gate_penalty_current = float(
            gate_penalty_start if gate_penalty_start is not None else gate_penalty_weight
        )
        self._gate_dropout_current = float(
            gate_dropout_start if gate_dropout_start is not None else gate_dropout_p
        )
        self._neighbor_context_scale_current = float(
            neighbor_context_scale_start if neighbor_context_scale_start is not None else neighbor_context_scale
        )
        self._gate_schedule_epochs = max(int(gate_schedule_epochs), 0)
        self._gate_penalty_start = (
            float(gate_penalty_start)
            if gate_penalty_start is not None else float(gate_penalty_weight)
        )
        self._gate_penalty_end = (
            float(gate_penalty_end) if gate_penalty_end is not None else float(gate_penalty_weight)
        )
        self._gate_dropout_start = (
            float(gate_dropout_start) if gate_dropout_start is not None else float(gate_dropout_p)
        )
        self._gate_dropout_end = (
            float(gate_dropout_end) if gate_dropout_end is not None else float(gate_dropout_p)
        )
        self._neighbor_context_scale_start = (
            float(neighbor_context_scale_start)
            if neighbor_context_scale_start is not None else float(neighbor_context_scale)
        )
        self._neighbor_context_scale_end = (
            float(neighbor_context_scale_end)
            if neighbor_context_scale_end is not None else float(neighbor_context_scale)
        )
        self.aux_neighbor_head_weight = float(max(aux_neighbor_head_weight, 0.0))
        self.aux_neighbor_head_weight_start = (
            float(aux_neighbor_head_weight_start)
            if aux_neighbor_head_weight_start is not None else self.aux_neighbor_head_weight
        )
        self.aux_neighbor_head_weight_end = (
            float(aux_neighbor_head_weight_end)
            if aux_neighbor_head_weight_end is not None else self.aux_neighbor_head_weight
        )
        self.aux_neighbor_head_schedule_epochs = max(int(aux_neighbor_head_schedule_epochs or 0), 0)
        self._aux_weight_current = self.aux_neighbor_head_weight_start
        self.aux_distance_temperature = float(max(aux_distance_temperature, 1e-6))
        
        # Fusion
        if fusion_method == "concat":
            fusion_input_dim = (
                self.tabular_encoder.output_dim
                + self.center_encoder.output_dim
                + (self.context_encoder.output_dim if self.context_encoder is not None else self.center_encoder.output_dim)
            )
            self.fusion = nn.Identity()
        elif fusion_method == "attention":
            self.fusion = AttentionFusion(
                self.tabular_encoder.output_dim,
                self.center_encoder.output_dim,
                fusion_dim
            )
            fusion_input_dim = fusion_dim
        elif fusion_method == "gated":
            self.fusion = GatedFusion(
                self.tabular_encoder.output_dim,
                self.center_encoder.output_dim,
                fusion_dim
            )
            fusion_input_dim = fusion_dim
        elif fusion_method == "film":
            self.fusion = FiLMFusion(
                self.tabular_encoder.output_dim,
                self.center_encoder.output_dim,
                fusion_dim
            )
            fusion_input_dim = fusion_dim
        elif fusion_method == "late_logit":
            self.fusion = None  # handled in forward
            fusion_input_dim = None
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Auxiliary neighbour-only head (optional)
        aux_input_dim = (
            self.context_encoder.output_dim
            if self.context_encoder is not None
            else self.center_encoder.output_dim
        )
        self.aux_head = None
        if self.aux_neighbor_head_weight > 0:
            aux_hidden = max(aux_input_dim // 2, 128)
            self.aux_head = nn.Sequential(
                nn.Linear(aux_input_dim, aux_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(aux_hidden, 1),
            )
        self._cached_aux_logits = None
        
        # Classifier (for early fusion modes)
        if fusion_method != "late_logit":
            out_dim = 1 if num_classes == 1 else num_classes
            self.classifier = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_dim // 2),
                nn.BatchNorm1d(fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim // 2, out_dim)
            )
        else:
            self.late_head = LateFusionHead(
                self.tabular_encoder.output_dim,
                self.center_encoder.output_dim
            )

        # -------- Loss & buffers ----------
        self.loss_type = loss_type
        # pos_weight gets updated from data at fit-start; initialize to global ratio estimate
        self.register_buffer("pos_weight", torch.tensor(38.594107, dtype=torch.float32))
        # logit bias init from prior (updated at fit-start as well)
        b0 = math.log(initial_pos_prior / max(1e-6, 1 - initial_pos_prior))
        self.register_buffer("logit_bias_init", torch.tensor(b0, dtype=torch.float32))

        # Best threshold (learned on validation)
        self.best_thr = 0.5

        # Set initial bias if early-fusion head exists
        if fusion_method != "late_logit":
            last_linear = self.classifier[-1]
            if isinstance(last_linear, nn.Linear) and last_linear.out_features == 1:
                with torch.no_grad():
                    last_linear.bias.data.fill_(self.logit_bias_init.item())

        # ---- Metrics (binary) ----
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_specificity = Specificity(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_auprc = AveragePrecision(task="binary")
        self.val_iou = JaccardIndex(task="binary")
        self.val_confusion_matrix = ConfusionMatrix(task="binary")
        self.test_acc = Accuracy(task="binary")
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")
        self.test_specificity = Specificity(task="binary")
        self.test_f1 = F1Score(task="binary")
        self.test_auroc = AUROC(task="binary")
        self.test_auprc = AveragePrecision(task="binary")
        self.test_iou = JaccardIndex(task="binary")
        self.test_confusion_matrix = ConfusionMatrix(task="binary")
        
        # Initialize best threshold for checkpointing
        self.best_thr = 0.5
    
    def _aggregate_knn_embeddings(self, batch):
        """
        Aggregate k nearest residue embeddings for SAS points.
        
        Supports two aggregation modes:
        - mean: Pre-aggregated embeddings from dataset (existing behavior)
        - transformer: Attention-based aggregation using NeighborAttentionEncoder
        
        Args:
            batch: Batch dictionary containing ESM embeddings and k-NN metadata
            
        Returns:
            ESM embeddings: [B, 2560] aggregated vector
        """
        # Detect aggregation mode from batch or use model default
        mode = batch.get('aggregation_mode', self.hparams.aggregation_mode)
        
        if mode == 'transformer':
            # ⭐ NEW: Transformer-based aggregation
            if 'esm_neighbors' not in batch:
                log.warning("Transformer mode requested but 'esm_neighbors' not in batch. "
                           "Falling back to pre-aggregated embedding.")
                return batch['esm']
            
            neighbors = batch['esm_neighbors']  # [B, k, 2560]
            distances = batch['neighbor_distances']  # [B, k]
            center = batch['esm']  # [B, 2560]
            
            # Create mask for valid neighbors (finite distances)
            mask = torch.isfinite(distances) & (distances < 1e6)
            # Clamp invalid distances to 30Å (far but finite) so distance bias can learn "invalid"
            # This is better than zero (which implies "at center") for attention bias learning
            distances = torch.where(mask, distances, torch.full_like(distances, 30.0))
            
            # Inject the center embedding as an additional neighbor (distance=0)
            center_expanded = center.unsqueeze(1)  # [B, 1, 2560]
            neighbors = torch.cat([center_expanded, neighbors], dim=1)  # [B, k+1, 2560]
            center_dist = torch.zeros(distances.size(0), 1, device=distances.device, dtype=distances.dtype)
            distances = torch.cat([center_dist, distances], dim=1)  # [B, k+1]
            center_mask = torch.ones(mask.size(0), 1, device=mask.device, dtype=torch.bool)
            mask = torch.cat([center_mask, mask], dim=1)  # [B, k+1]
            
            # Apply attention-based aggregation (now includes center residue)
            collect_attention_stats = not self.training
            encoder_output = self.neighbor_encoder(
                neighbors=neighbors,
                distances=distances,
                mask=mask,
                return_attention=collect_attention_stats
            )
            if collect_attention_stats:
                neighbor_context, attn_stats = encoder_output
            else:
                neighbor_context = encoder_output
                attn_stats = None
            
            # ⭐ CRITICAL: Learned per-sample gate between centre & neighbour context
            centre_dropped = self.center_dropout(center)
            centre_normed = self.center_norm(centre_dropped)
            neighbor_normed = self.neighbor_norm(neighbor_context)
            gate_input = torch.cat([centre_normed, neighbor_normed], dim=-1)  # [B, 5120]
            gate_logits = self.gate_mlp(gate_input).squeeze(-1)  # [B]
            temperature = float(max(self.hparams.gate_temperature, 1e-3))
            gate_logits = gate_logits / temperature
            alpha = torch.sigmoid(gate_logits)
            dropout_rate = self._gate_dropout_current
            if self.training and dropout_rate > 0:
                drop_mask = (torch.rand_like(alpha) < dropout_rate)
                alpha = torch.where(
                    drop_mask,
                    torch.full_like(alpha, float(self.hparams.gate_dropout_target)),
                    alpha
                )
                self._last_gate_dropout = drop_mask.float().mean().detach()
            else:
                self._last_gate_dropout = None
            alpha_unsq = alpha.unsqueeze(-1)
            neighbor_context_blend = self.context_blend_norm(neighbor_context)
            neighbor_context_blend = neighbor_context_blend * self._neighbor_context_scale_current
            blended = alpha_unsq * center + (1 - alpha_unsq) * neighbor_context_blend  # [B, 2560]
            if self.training:
                self._last_alpha_mean = alpha.mean().detach()
            
            # Expose both the original centre embedding and the blended context downstream
            # For validation logging
            if not self.training and hasattr(self, "_val_alpha_buffer"):
                self._val_alpha_buffer.append(alpha.detach())
                self._val_neighbor_fraction.append(mask[:, 1:].float().mean(dim=1).detach())
                centre_norm_mag = center.norm(dim=-1).detach()
                neighbor_norm_mag = neighbor_context_blend.norm(dim=-1).detach()
                self._val_center_norm.append(centre_norm_mag)
                self._val_neighbor_norm.append(neighbor_norm_mag)
                if attn_stats and "entropy_per_head" in attn_stats:
                    self._val_attn_entropy.append(attn_stats["entropy_per_head"].detach().cpu())
                if attn_stats and "entropy_per_layer" in attn_stats:
                    self._val_attn_entropy_layer.append(attn_stats["entropy_per_layer"].detach().cpu())

            if self.aux_neighbor_head_weight > 0:
                neigh_mask = mask[:, 1:]
                neigh_dist = distances[:, 1:]
                if neigh_dist.numel() > 0:
                    filled = torch.where(neigh_mask, neigh_dist, torch.full_like(neigh_dist, 30.0))
                    min_dist = filled.min(dim=1).values
                else:
                    min_dist = torch.full((neighbors.size(0),), 30.0, device=neighbors.device)
                self._cached_neighbor_metrics["min_dist"] = min_dist.detach()
            else:
                self._cached_neighbor_metrics.pop("min_dist", None)
            
            return center, blended  # both [B, 2560]
        
        elif mode == 'mean':
            # Existing behavior: embeddings already aggregated by dataset
            centre = batch['esm']
            self._cached_neighbor_metrics.pop("min_dist", None)
            return centre, centre  # Treat centre = context
        
        else:
            raise ValueError(f"Unknown aggregation_mode: {mode}. Expected 'mean' or 'transformer'.")
        
        # NOTE: The old approach tried to aggregate here but lacked access to full H5 tensor.
        # The new approach (dataset-level aggregation) is:
        # 1. More efficient (no redundant operations)
        # 2. Cleaner (separation of concerns)
        # 3. Correct (has access to full ESM tensor via shared memory)
        #
        # If you need to verify aggregation happened, check batch.get('knn_aggregated', False)
        
        # For debugging/logging, you can check if aggregation was successful:
        # if self.hparams.k_res_neighbors > 1:
        #     if batch.get('knn_aggregated', False):
        #         log.debug(f"Using k-NN aggregated embeddings (k={self.hparams.k_res_neighbors})")
        #     else:
        #         log.debug(f"k-NN aggregation not available, using original embeddings")
    
    
    def forward(self, tabular_feat, esm_feat=None, batch=None):
        """
        Forward pass with optional k-NN ESM aggregation.
        
        Args:
            tabular_feat: Tabular features
            esm_feat: ESM features (legacy support)
            batch: Full batch dictionary (for k-NN aggregation)
        """
        # Handle both legacy and new calling conventions
        if batch is not None:
            # New calling convention with full batch
            tabular_feat = batch['tabular']
            center_raw, context_raw = self._aggregate_knn_embeddings(batch)
        else:
            if esm_feat is None:
                raise ValueError("Either esm_feat or batch must be provided")
            center_raw, context_raw = (esm_feat, esm_feat)
        
        # Optional modality dropout (training only)
        if self.training and self.hparams.modality_dropout_p > 0:
            p = float(self.hparams.modality_dropout_p)
            if torch.rand(1).item() < p:  # drop tabular
                tabular_feat = torch.zeros_like(tabular_feat)
            if torch.rand(1).item() < p:  # drop ESM embeddings
                center_raw = torch.zeros_like(center_raw)
                context_raw = torch.zeros_like(context_raw)

        h_t = self.tabular_encoder(tabular_feat)
        h_center = self.center_encoder(center_raw)
        if self.context_encoder is not None:
            h_context = self.context_encoder(context_raw)
        else:
            h_context = self.center_encoder(context_raw)

        if self.hparams.fusion_method == "concat":
            fused = torch.cat([h_t, h_center, h_context], dim=-1)
            logits = self.classifier(fused).squeeze(-1)
            if self.aux_head is not None:
                self._cached_aux_logits = self.aux_head(h_context).squeeze(-1)
            else:
                self._cached_aux_logits = None
        elif self.hparams.fusion_method in ("gated", "attention", "film"):
            raise NotImplementedError("Transformer aggregation currently supports fusion_method='concat' only")
        elif self.hparams.fusion_method == "late_logit":
            raise NotImplementedError("Transformer aggregation currently supports fusion_method='concat' only")
        else:
            raise RuntimeError(f"Unknown fusion method: {self.hparams.fusion_method}")

        return logits  # [B] single logit

    def _shared_step(self, batch, batch_idx):
        tabular = batch['tabular']
        esm = batch['esm']
        labels = batch['label'].float()  # BCE needs float targets (0./1.)
        
        # Use new forward method with batch for k-NN support
        if self.hparams.k_res_neighbors > 1:
            logits = self(tabular, batch=batch)  # Use batch for k-NN aggregation
        else:
            logits = self(tabular, esm)          # Legacy method for k=1
        aux_logits = getattr(self, "_cached_aux_logits", None)
        self._cached_aux_logits = None
        
        # Use proper loss functions based on configuration
        if self.loss_type == "bce":
            criterion = BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        elif self.loss_type == "focal":
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
        elif self.loss_type == "asl":
            criterion = AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05)
        else:
            # Default fallback to BCE
            criterion = BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        
        loss = criterion(logits, labels)
        if (
            self.training
            and self._aux_weight_current > 0
            and aux_logits is not None
        ):
            aux_loss_raw = F.binary_cross_entropy_with_logits(aux_logits, labels, reduction="none")
            weights = torch.ones_like(aux_loss_raw)
            min_dist = self._cached_neighbor_metrics.pop("min_dist", None)
            if min_dist is not None:
                weights = torch.exp(-min_dist / self.aux_distance_temperature)
                weights = weights.detach()
                weights = weights / weights.mean().clamp_min(1e-6)
            aux_loss = (weights * aux_loss_raw).mean()
            loss = loss + self._aux_weight_current * aux_loss
            self._last_aux_loss = aux_loss.detach()
            self.log(
                "neighbors/aux_loss",
                aux_loss.detach(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=labels.size(0),
            )
            self.log(
                "neighbors/aux_weight",
                float(self._aux_weight_current),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=labels.size(0),
            )
        else:
            self._cached_neighbor_metrics.pop("min_dist", None)
            self._last_aux_loss = None
        if self.training and self._last_alpha_mean is not None:
            weight = self._gate_penalty_current
            target = float(self.hparams.gate_penalty_target)
            alpha_penalty = weight * (self._last_alpha_mean - target).abs()
            loss = loss + alpha_penalty
            self.log(
                "regularizer/alpha_penalty",
                alpha_penalty,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=labels.size(0),
            )
        if self.training and getattr(self, "_last_gate_dropout", None) is not None:
            self.log(
                "regularizer/gate_dropout_rate",
                self._last_gate_dropout,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=labels.size(0),
            )
        probs = torch.sigmoid(logits)
        return loss, logits, probs, labels

    def training_step(self, batch, batch_idx):
        loss, logits, probs, labels = self._shared_step(batch, batch_idx)
        preds = (probs >= 0.5).long()  # training-only metric
        batch_size = labels.size(0)
        self.train_acc(preds, labels.long())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def on_validation_epoch_start(self):
        self._val_probs, self._val_labels = [], []
        self._val_alpha_buffer = []
        self._val_neighbor_fraction = []
        self._val_center_norm = []
        self._val_neighbor_norm = []
        self._val_attn_entropy = []
        self._val_attn_entropy_layer = []
        self._last_attn_entropy_mean = None
        self._last_aux_loss = None
        self._cached_neighbor_metrics = {}

    def validation_step(self, batch, batch_idx):
        loss, logits, probs, labels = self._shared_step(batch, batch_idx)
        
        # Aggregate basic metrics that are safe with sparse positives
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=labels.size(0),
        )
        
        # Accumulate predictions/labels for epoch-level metric computation
        self._val_probs.append(probs.detach())
        self._val_labels.append(labels.detach())

    def on_validation_epoch_end(self):
        if not hasattr(self, '_val_probs') or len(self._val_probs) == 0:
            return
            
        p = torch.cat(self._val_probs)
        y = torch.cat(self._val_labels).long()
        
        # DDP-safe gather with all_gather_object
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            ws = torch.distributed.get_world_size()
            
            # Gather predictions and labels from all ranks
            plist = [None] * ws
            ylist = [None] * ws
            torch.distributed.all_gather_object(plist, p.cpu().numpy())
            torch.distributed.all_gather_object(ylist, y.cpu().numpy())
            
            # Concatenate gathered data
            p = torch.from_numpy(np.concatenate(plist)).to(self.device)
            y = torch.from_numpy(np.concatenate(ylist)).long().to(self.device)

        thrs = torch.linspace(0.01, 0.99, 50, device=p.device)
        best_iou, best_thr = 0.0, 0.5
        for t in thrs:
            preds = (p >= t).long()
            inter = ((preds==1) & (y==1)).sum().item()
            union = ((preds==1) | (y==1)).sum().item()
            iou = 0.0 if union==0 else inter/union
            if iou > best_iou:
                best_iou, best_thr = iou, float(t)
        
        # Broadcast best threshold to all ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            t = torch.tensor([best_thr], dtype=torch.float32, device=self.device)
            torch.distributed.broadcast(t, src=0)
            best_thr = float(t.item())
        
        self.best_thr = best_thr
        batch_size = len(p)  # Total validation set size
        
        # Compute epoch-level metrics using the optimized threshold
        preds = (p >= best_thr).long()
        total = torch.tensor(float(len(p)), device=self.device)
        tp = ((preds == 1) & (y == 1)).sum().float()
        tn = ((preds == 0) & (y == 0)).sum().float()
        fp = ((preds == 1) & (y == 0)).sum().float()
        fn = ((preds == 0) & (y == 1)).sum().float()
        eps = torch.finfo(torch.float32).eps
        
        acc = (tp + tn) / torch.clamp(total, min=1.0)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        f1 = (2 * precision * recall) / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        
        # Epoch-level AUROC and AUPRC from full predictions
        auroc_val = tm_auroc(p, y, task="binary")
        auprc_val = tm_average_precision(p, y, task="binary")
        
        metric_tensors = {
            "val/auroc": auroc_val,
            "val/auprc": auprc_val,
            "val/acc": acc,
            "val/precision": precision,
            "val/recall": recall,
            "val/specificity": specificity,
            "val/f1": f1,
            "val/iou": iou,
            "val/iou_opt": torch.tensor(best_iou, device=self.device, dtype=torch.float32),
            "val/threshold_opt": torch.tensor(best_thr, device=self.device, dtype=torch.float32),
        }
        
        for name, value in metric_tensors.items():
            self.log(
                name,
                value,
                prog_bar=name in {"val/auprc", "val/auroc", "val/f1"},
                sync_dist=True,
                batch_size=batch_size,
            )
        
        alpha_mean = None
        if hasattr(self, "_val_alpha_buffer") and len(self._val_alpha_buffer) > 0:
            alpha_tensor = torch.cat([a.view(-1) for a in self._val_alpha_buffer])
            alpha_mean = alpha_tensor.mean()
            alpha_std = alpha_tensor.std(unbiased=False)
            self.log(
                "residual/alpha",
                alpha_mean,
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )
            self.log(
                "residual/alpha_std",
                alpha_std,
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )
            self._val_alpha_buffer = []
        
        if hasattr(self, "_val_neighbor_fraction") and len(self._val_neighbor_fraction) > 0:
            frac_tensor = torch.cat([f.view(-1) for f in self._val_neighbor_fraction])
            frac_mean = frac_tensor.mean()
            self.log(
                "neighbors/valid_fraction",
                frac_mean,
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )
            self._val_neighbor_fraction = []
        
        if hasattr(self, "_val_center_norm") and len(self._val_center_norm) > 0:
            center_norm_tensor = torch.cat([c.view(-1) for c in self._val_center_norm])
            neighbor_norm_tensor = torch.cat([n.view(-1) for n in self._val_neighbor_norm])
            self.log(
                "neighbors/center_norm_mean",
                center_norm_tensor.mean(),
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )
            self.log(
                "neighbors/context_norm_mean",
                neighbor_norm_tensor.mean(),
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )
            self.log(
                "neighbors/context_norm_std",
                neighbor_norm_tensor.std(unbiased=False),
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )
            self._val_center_norm = []
            self._val_neighbor_norm = []
        
        if hasattr(self, "_val_attn_entropy") and len(self._val_attn_entropy) > 0:
            entropy_tensor = torch.stack([e.to(self.device) for e in self._val_attn_entropy])
            mean_entropy = entropy_tensor.mean(dim=0)  # [num_heads]
            self._last_attn_entropy_mean = mean_entropy.mean().detach()
            self.log(
                "neighbors/attn_entropy_mean",
                mean_entropy.mean(),
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )
            for head_idx in range(mean_entropy.numel()):
                self.log(
                    f"neighbors/attn_entropy_head{head_idx}",
                    mean_entropy[head_idx],
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=batch_size,
                )
            self._val_attn_entropy = []
        
        if hasattr(self, "_val_attn_entropy_layer") and len(self._val_attn_entropy_layer) > 0:
            layer_entropy_tensor = torch.stack([e.to(self.device) for e in self._val_attn_entropy_layer])
            mean_layer_entropy = layer_entropy_tensor.mean(dim=0)  # [num_layers, num_heads]
            for layer_idx in range(mean_layer_entropy.size(0)):
                self.log(
                    f"neighbors/attn_entropy_layer{layer_idx}",
                    mean_layer_entropy[layer_idx].mean(),
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=batch_size,
                )
            self._val_attn_entropy_layer = []
        
        if alpha_mean is not None:
            self.log(
                "neighbors/alpha_median",
                alpha_tensor.median(),
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )
            self.log(
                "neighbors/alpha_min",
                alpha_tensor.min(),
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )
            self.log(
                "neighbors/alpha_max",
                alpha_tensor.max(),
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )
        
        # Console summary on rank 0 for easier monitoring
        if getattr(self.trainer, "is_global_zero", False):
            train_loss = self.trainer.callback_metrics.get("train/loss_epoch")
            val_loss = self.trainer.callback_metrics.get("val/loss")
            parts = [f"[epoch {self.current_epoch:03d}]"]
            if train_loss is not None:
                parts.append(f"train_loss={float(train_loss):.4f}")
            if val_loss is not None:
                parts.append(f"val_loss={float(val_loss):.4f}")
            parts.append(f"val_auprc={float(auprc_val):.4f}")
            parts.append(f"val_auroc={float(auroc_val):.4f}")
            parts.append(f"val_f1={float(f1):.4f}")
            if alpha_mean is not None:
                parts.append(f"alpha={float(alpha_mean):.3f}")
            parts.append(f"gate_w={self._gate_penalty_current:.3f}")
            parts.append(f"gate_do={self._gate_dropout_current:.2f}")
            parts.append(f"ctx_scale={self._neighbor_context_scale_current:.2f}")
            if self._last_attn_entropy_mean is not None:
                parts.append(f"entropy={float(self._last_attn_entropy_mean):.3f}")
            aux_epoch = self.trainer.callback_metrics.get("neighbors/aux_loss_epoch")
            if aux_epoch is not None:
                parts.append(f"aux={float(aux_epoch):.3f}")
            self.print(" ".join(parts))

    def on_test_epoch_start(self):
        self._test_probs, self._test_labels = [], []

    def test_step(self, batch, batch_idx):
        loss, logits, probs, labels = self._shared_step(batch, batch_idx)
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=labels.size(0),
        )
        
        self._test_probs.append(probs.detach())
        self._test_labels.append(labels.detach())
        return loss

    def on_test_epoch_end(self):
        if not hasattr(self, "_test_probs") or len(self._test_probs) == 0:
            return
        
        p = torch.cat(self._test_probs)
        y = torch.cat(self._test_labels).long()
        
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            ws = torch.distributed.get_world_size()
            plist = [None] * ws
            ylist = [None] * ws
            torch.distributed.all_gather_object(plist, p.cpu().numpy())
            torch.distributed.all_gather_object(ylist, y.cpu().numpy())
            p = torch.from_numpy(np.concatenate(plist)).to(self.device)
            y = torch.from_numpy(np.concatenate(ylist)).long().to(self.device)
        
        batch_size = len(p)
        best_thr = getattr(self, "best_thr", 0.5)
        preds = (p >= best_thr).long()
        
        tp = ((preds == 1) & (y == 1)).sum().float()
        tn = ((preds == 0) & (y == 0)).sum().float()
        fp = ((preds == 1) & (y == 0)).sum().float()
        fn = ((preds == 0) & (y == 1)).sum().float()
        eps = torch.finfo(torch.float32).eps
        total = torch.tensor(float(len(p)), device=self.device)
        
        acc = (tp + tn) / torch.clamp(total, min=1.0)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        specificity = tn / (tn + fp + eps)
        f1 = (2 * precision * recall) / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        
        auroc_val = tm_auroc(p, y, task="binary")
        auprc_val = tm_average_precision(p, y, task="binary")
        
        metric_tensors = {
            "test/auroc": auroc_val,
            "test/auprc": auprc_val,
            "test/acc": acc,
            "test/precision": precision,
            "test/recall": recall,
            "test/specificity": specificity,
            "test/f1": f1,
            "test/iou": iou,
        }
        
        for name, value in metric_tensors.items():
            self.log(
                name,
                value,
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )

    def configure_optimizers(self):
        opt = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is None:
            return opt

        # Prefer a user-provided total_steps; otherwise use Lightning's estimate
        total_steps = self.hparams.scheduler.keywords.get("total_steps")
        if total_steps is None:
            # Available in PL 2.x at configure_optimizers time
            est = getattr(self.trainer, "estimated_stepping_batches", None)
            if est is None or est <= 0:
                # Final fallback if something unexpected happens
                # (better to over-estimate than under-estimate)
                if self.trainer.num_training_batches:
                    steps_per_epoch = math.ceil(
                        int(self.trainer.num_training_batches) /
                        max(1, int(self.trainer.accumulate_grad_batches or 1))
                    )
                    est = steps_per_epoch * int(self.trainer.max_epochs)
                else:
                    est = int(self.trainer.max_epochs) * 1000  # safe overestimate
            total_steps = int(est)
            self.print(f"✅ OneCycleLR total_steps set to {total_steps}")

        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.hparams.scheduler.keywords["max_lr"],
            total_steps=total_steps,                      # <— key change
            pct_start=self.hparams.scheduler.keywords.get("pct_start", 0.3),
            anneal_strategy=self.hparams.scheduler.keywords.get("anneal_strategy", "cos"),
            div_factor=self.hparams.scheduler.keywords.get("div_factor", 25.0),
            final_div_factor=self.hparams.scheduler.keywords.get("final_div_factor", 1e4),
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self._gate_schedule_epochs > 0:
            epoch = self.current_epoch
            ratio = min(max(epoch, 0) / float(self._gate_schedule_epochs), 1.0)
            self._gate_penalty_current = (
                self._gate_penalty_start
                + (self._gate_penalty_end - self._gate_penalty_start) * ratio
            )
            self._gate_dropout_current = (
                self._gate_dropout_start
                + (self._gate_dropout_end - self._gate_dropout_start) * ratio
            )
            self._neighbor_context_scale_current = (
                self._neighbor_context_scale_start
                + (self._neighbor_context_scale_end - self._neighbor_context_scale_start) * ratio
            )
            self.log(
                "regularizer/gate_penalty_weight",
                float(self._gate_penalty_current),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "regularizer/gate_dropout_target",
                float(self._gate_dropout_current),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "neighbors/context_scale",
                float(self._neighbor_context_scale_current),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        else:
            self._gate_penalty_current = float(self._gate_penalty_start)
            self._gate_dropout_current = float(self._gate_dropout_start)
            self._neighbor_context_scale_current = float(self._neighbor_context_scale_start)

        if self.aux_neighbor_head_schedule_epochs > 0:
            epoch = self.current_epoch
            ratio = min(max(epoch, 0) / float(self.aux_neighbor_head_schedule_epochs), 1.0)
            self._aux_weight_current = (
                self.aux_neighbor_head_weight_start
                + (self.aux_neighbor_head_weight_end - self.aux_neighbor_head_weight_start) * ratio
            )
        else:
            self._aux_weight_current = self.aux_neighbor_head_weight_start

        self.log(
            "neighbors/aux_weight",
            float(self._aux_weight_current),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )


    def on_save_checkpoint(self, checkpoint):
        """Save best threshold in checkpoint for persistence."""
        checkpoint["best_thr"] = getattr(self, "best_thr", 0.5)
        
    def on_load_checkpoint(self, checkpoint):
        """Load best threshold from checkpoint."""
        self.best_thr = float(checkpoint.get("best_thr", 0.5))

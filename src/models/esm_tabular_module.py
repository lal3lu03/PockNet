"""
ESM + Tabular Feature Fusion Model for Binding Site Prediction.

This model combines:
1. Tabular features (45D) from physicochemical properties
2. ESM2 embeddings (2560D) with proper residue-specific mapping
3. Fusion strategies for optimal performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision
from typing import Dict, Any, Optional, List
import logging

log = logging.getLogger(__name__)


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


class EsmTabularModule(LightningModule):
    """Lightning module for ESM + Tabular feature fusion."""
    
    def __init__(
        self,
        tabular_dim: int = 45,
        esm_dim: int = 2560,
        hidden_dims: List[int] = [1024, 512, 256],
        dropout: float = 0.2,
        fusion_method: str = "concat",
        fusion_dim: int = 512,
        num_classes: int = 2,
        optimizer: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Encoders
        self.tabular_encoder = TabularEncoder(tabular_dim, hidden_dims, dropout)
        self.esm_encoder = EsmEncoder(esm_dim, hidden_dims, dropout)
        
        # Fusion
        if fusion_method == "concat":
            fusion_input_dim = self.tabular_encoder.output_dim + self.esm_encoder.output_dim
            self.fusion = nn.Identity()
        elif fusion_method == "attention":
            self.fusion = AttentionFusion(
                self.tabular_encoder.output_dim, 
                self.esm_encoder.output_dim, 
                fusion_dim
            )
            fusion_input_dim = fusion_dim
        elif fusion_method == "gated":
            self.fusion = GatedFusion(
                self.tabular_encoder.output_dim, 
                self.esm_encoder.output_dim, 
                fusion_dim
            )
            fusion_input_dim = fusion_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        # Metrics
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")
        
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_auprc = AveragePrecision(task="binary")
        
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")
        self.test_f1 = F1Score(task="binary")
        self.test_auroc = AUROC(task="binary")
        self.test_auprc = AveragePrecision(task="binary")
    
    def forward(self, tabular_feat, esm_feat):
        # Encode features
        tab_encoded = self.tabular_encoder(tabular_feat)
        esm_encoded = self.esm_encoder(esm_feat)
        
        # Fusion
        if self.hparams.fusion_method == "concat":
            fused = torch.cat([tab_encoded, esm_encoded], dim=-1)
        else:
            fused = self.fusion(tab_encoded, esm_encoded)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits
    
    def _shared_step(self, batch, batch_idx):
        tabular = batch['tabular']
        esm = batch['esm']
        labels = batch['label']
        
        logits = self(tabular, esm)
        loss = F.cross_entropy(logits, labels)
        probs = F.softmax(logits, dim=-1)[:, 1]  # Probability of positive class
        
        return loss, logits, probs, labels
    
    def training_step(self, batch, batch_idx):
        loss, logits, probs, labels = self._shared_step(batch, batch_idx)
        
        # Metrics
        preds = torch.argmax(logits, dim=-1)
        self.train_acc(preds, labels)
        
        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, logits, probs, labels = self._shared_step(batch, batch_idx)
        
        # Metrics
        preds = torch.argmax(logits, dim=-1)
        self.val_acc(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)
        self.val_auroc(probs, labels)
        self.val_auprc(probs, labels)
        
        # Logging
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, logits, probs, labels = self._shared_step(batch, batch_idx)
        
        # Metrics
        preds = torch.argmax(logits, dim=-1)
        self.test_acc(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_f1(preds, labels)
        self.test_auroc(probs, labels)
        self.test_auprc(probs, labels)
        
        # Logging
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test/auroc", self.test_auroc, on_step=False, on_epoch=True)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = self.hparams.optimizer(params=self.parameters())
        
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/f1",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        
        return optimizer

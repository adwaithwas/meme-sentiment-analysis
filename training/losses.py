"""
Multi-task loss with class weighting and optional learnable task weights.
"""

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """
    Combines losses from multiple task heads:
    - Sentiment: CrossEntropyLoss (3 classes)
    - Humor: CrossEntropyLoss (4 classes)
    - Sarcasm: CrossEntropyLoss (2 classes, binary)
    
    Supports:
    - Per-class weights (for imbalanced datasets)
    - Learnable task weights (uncertainty-based, Kendall et al. 2018)
    """
    
    def __init__(self, config, sentiment_weights=None, humor_weights=None, sarcasm_weights=None):
        super().__init__()
        
        # Per-task loss functions with optional class weights
        self.sentiment_loss = nn.CrossEntropyLoss(
            weight=sentiment_weights.to(config.device) if sentiment_weights is not None else None
        )
        self.humor_loss = nn.CrossEntropyLoss(
            weight=humor_weights.to(config.device) if humor_weights is not None else None
        )
        self.sarcasm_loss = nn.CrossEntropyLoss(
            weight=sarcasm_weights.to(config.device) if sarcasm_weights is not None else None
        )
        
        # Fixed task weights from config
        self.fixed_weights = {
            'sentiment': config.sentiment_loss_weight,
            'humor': config.humor_loss_weight,
            'sarcasm': config.sarcasm_loss_weight,
        }
        
        # Learnable task weights (log variance parameterization)
        # Higher uncertainty → lower weight (automatic balancing)
        self.use_learnable = config.use_learnable_task_weights
        if self.use_learnable:
            self.log_vars = nn.ParameterDict({
                'sentiment': nn.Parameter(torch.zeros(1)),
                'humor': nn.Parameter(torch.zeros(1)),
                'sarcasm': nn.Parameter(torch.zeros(1)),
            })
    
    def forward(self, outputs, labels):
        """
        Compute combined multi-task loss.
        
        Args:
            outputs: dict with keys *_logits
            labels: dict with keys sentiment, humor, sarcasm
        Returns:
            total_loss: scalar tensor
        """
        losses = {
            'sentiment': self.sentiment_loss(outputs['sentiment_logits'], labels['sentiment']),
            'humor': self.humor_loss(outputs['humor_logits'], labels['humor']),
            'sarcasm': self.sarcasm_loss(outputs['sarcasm_logits'], labels['sarcasm']),
        }
        
        if self.use_learnable:
            # Uncertainty-based weighting: L_total = Σ (1/(2σ²)) * L_i + log(σ)
            total = 0
            for task, loss in losses.items():
                precision = torch.exp(-self.log_vars[task])
                total += precision * loss + self.log_vars[task]
            return total.squeeze()
        else:
            # Fixed weighting
            total = sum(self.fixed_weights[t] * l for t, l in losses.items())
            return total

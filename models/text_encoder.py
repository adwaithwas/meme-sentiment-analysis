"""
Text Encoder: XLM-RoBERTa wrapper for multilingual + Hinglish support.
Extracts [CLS] token embedding and projects to fusion dimension.
"""

import torch
import torch.nn as nn
from transformers import XLMRobertaModel


class TextEncoder(nn.Module):
    """
    XLM-RoBERTa text encoder with selective layer freezing.
    
    - Loads pretrained xlm-roberta-base (768-dim, 12 layers)
    - Freezes first N layers for stable fine-tuning
    - Extracts [CLS] token as sentence representation
    - Projects 768-dim → fusion_dim via linear layer
    """
    
    def __init__(self, model_name='xlm-roberta-base', fusion_dim=512,
                 freeze_layers=8, dropout=0.1):
        super().__init__()
        
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size  # 768
        
        # Freeze embedding layer and first N transformer layers
        # This prevents catastrophic forgetting of pretrained knowledge
        self._freeze_layers(freeze_layers)
        
        # Projection: 768 → fusion_dim
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def _freeze_layers(self, num_layers):
        """Freeze embeddings + first num_layers transformer layers."""
        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze first N encoder layers
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch, seq_len] tokenized text
            attention_mask: [batch, seq_len] attention mask
        Returns:
            text_features: [batch, fusion_dim] projected text embedding
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token (first token) as sentence representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        
        # Project to fusion dimension
        text_features = self.projection(cls_output)  # [batch, fusion_dim]
        
        return text_features

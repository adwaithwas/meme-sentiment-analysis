"""
Full Multimodal Meme Analyzer: combines text encoder, image encoder,
fusion module, and multi-task prediction heads.
"""

import torch
import torch.nn as nn
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.fusion import CrossAttentionFusion, ConcatFusion


class MultimodalMemeAnalyzer(nn.Module):
    """
    End-to-end multimodal meme analysis model.
    
    Architecture:
        Image → ResNet50 → projection → ┐
                                         ├→ Cross-Attention Fusion → Task Heads
        Text → XLM-RoBERTa → projection → ┘
        
    Task Heads:
        - Sentiment: 3-class (positive, negative, neutral)
        - Humor: 4-class (not funny → hilarious)
        - Sarcasm: binary (sarcastic or not)
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # --- Encoders ---
        self.text_encoder = TextEncoder(
            model_name=config.text_model_name,
            fusion_dim=config.fusion_dim,
            freeze_layers=config.text_freeze_layers,
            dropout=config.fusion_dropout,
        )
        
        self.image_encoder = ImageEncoder(
            fusion_dim=config.fusion_dim,
            freeze_blocks=config.image_freeze_blocks,
            dropout=config.fusion_dropout,
        )
        
        # --- Fusion ---
        if config.fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                embed_dim=config.fusion_dim,
                num_heads=config.num_attention_heads,
                dropout=config.fusion_dropout,
            )
        else:
            self.fusion = ConcatFusion(
                embed_dim=config.fusion_dim,
                dropout=config.fusion_dropout,
            )
        
        # --- Multi-Task Prediction Heads ---
        # Each head: Linear → ReLU → Dropout → Linear → output
        self.sentiment_head = self._make_head(config.fusion_dim, config.num_sentiment_classes)
        self.humor_head = self._make_head(config.fusion_dim, config.num_humor_classes)
        self.sarcasm_head = self._make_head(config.fusion_dim, config.num_sarcasm_classes)
    
    def _make_head(self, input_dim, num_classes):
        """Create a 2-layer MLP classification head."""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.fusion_dropout),
            nn.Linear(input_dim // 2, num_classes),
        )
    
    def forward(self, input_ids, attention_mask, images):
        """
        Forward pass through the full model.
        
        Args:
            input_ids: [batch, seq_len] tokenized text
            attention_mask: [batch, seq_len]
            images: [batch, 3, 224, 224] normalized images
            
        Returns:
            dict with keys:
                sentiment_logits: [batch, 3]
                humor_logits: [batch, 4]
                sarcasm_logits: [batch, 2]
        """
        # Extract features from each modality
        text_features = self.text_encoder(input_ids, attention_mask)   # [B, fusion_dim]
        image_features = self.image_encoder(images)                    # [B, fusion_dim]
        
        # Fuse modalities
        fused = self.fusion(text_features, image_features)  # [B, fusion_dim]
        
        # Multi-task predictions
        return {
            'sentiment_logits': self.sentiment_head(fused),  # [B, 3]
            'humor_logits': self.humor_head(fused),          # [B, 4]
            'sarcasm_logits': self.sarcasm_head(fused),      # [B, 2]
        }
    
    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
            'total_M': f"{total / 1e6:.1f}M",
            'trainable_M': f"{trainable / 1e6:.1f}M",
        }

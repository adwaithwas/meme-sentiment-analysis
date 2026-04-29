"""
Multimodal Fusion modules: Cross-Attention and Concatenation.
Combines text and image features into a unified representation.
"""

import torch
import torch.nn as nn
import math


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion: text attends to image and image attends to text.
    
    This is superior to simple concatenation because it allows the model
    to learn which parts of one modality are relevant to the other.
    For memes, this captures relationships like "ironic text + contradicting image".
    
    Architecture:
        1. Text-to-Image attention (text queries, image keys/values)
        2. Image-to-Text attention (image queries, text keys/values)
        3. Combine via gated fusion
        4. MLP projection to final dimension
    """
    
    def __init__(self, embed_dim=512, num_heads=4, dropout=0.3):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Cross-attention: text attends to image
        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        
        # Cross-attention: image attends to text
        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        
        # Layer norms for residual connections
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Gated fusion: learn how much to weight each modality
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        # Final projection MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        
        self.final_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, text_features, image_features):
        """
        Args:
            text_features: [batch, embed_dim]
            image_features: [batch, embed_dim]
        Returns:
            fused: [batch, embed_dim] fused multimodal representation
        """
        # Reshape to [batch, 1, embed_dim] for attention (single token per modality)
        text_seq = text_features.unsqueeze(1)    # [B, 1, D]
        image_seq = image_features.unsqueeze(1)  # [B, 1, D]
        
        # Text-to-Image attention: text queries, image keys/values
        t2i, _ = self.text_to_image_attn(text_seq, image_seq, image_seq)
        text_attended = self.norm1(text_seq + t2i).squeeze(1)  # [B, D]
        
        # Image-to-Text attention: image queries, text keys/values
        i2t, _ = self.image_to_text_attn(image_seq, text_seq, text_seq)
        image_attended = self.norm2(image_seq + i2t).squeeze(1)  # [B, D]
        
        # Gated fusion: learn optimal modality weighting
        combined = torch.cat([text_attended, image_attended], dim=1)  # [B, 2D]
        gate_values = self.gate(combined)  # [B, D] — values in [0,1]
        
        fused = gate_values * text_attended + (1 - gate_values) * image_attended  # [B, D]
        
        # Final MLP with residual
        fused = self.final_norm(fused + self.mlp(fused))  # [B, D]
        
        return fused


class ConcatFusion(nn.Module):
    """
    Simple concatenation fusion (baseline).
    Concatenates text and image features, then projects down.
    """
    
    def __init__(self, embed_dim=512, dropout=0.3):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, text_features, image_features):
        """
        Args:
            text_features: [batch, embed_dim]
            image_features: [batch, embed_dim]
        Returns:
            fused: [batch, embed_dim]
        """
        combined = torch.cat([text_features, image_features], dim=1)
        return self.mlp(combined)

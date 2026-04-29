"""
Image Encoder: ResNet50 wrapper for visual feature extraction.
Removes classification head, projects features to fusion dimension.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ImageEncoder(nn.Module):
    """
    ResNet50 image encoder with selective block freezing.
    
    - Loads ImageNet-pretrained ResNet50 (2048-dim output)
    - Removes final classification FC layer
    - Freezes early conv blocks for stable fine-tuning
    - Projects 2048-dim → fusion_dim via linear layer
    """
    
    def __init__(self, fusion_dim=512, freeze_blocks=6, dropout=0.1):
        super().__init__()
        
        # Load pretrained ResNet50
        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.hidden_size = 2048  # ResNet50 final feature dim
        
        # Remove the final FC layer — we want features, not ImageNet classes
        # ResNet50 structure: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,  # Block 1-2
            base_model.layer2,  # Block 3-4
            base_model.layer3,  # Block 5-6
            base_model.layer4,  # Block 7-8
        )
        self.avgpool = base_model.avgpool
        
        # Freeze early layers for stable training
        self._freeze_blocks(freeze_blocks)
        
        # Projection: 2048 → fusion_dim
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def _freeze_blocks(self, num_blocks):
        """
        Freeze early ResNet blocks. The model has ~8 major blocks:
        conv1(1), layer1(2), layer2(2), layer3(2), layer4(0-2 depending on setting)
        """
        all_blocks = list(self.features.children())
        for i, block in enumerate(all_blocks):
            if i < num_blocks:
                for param in block.parameters():
                    param.requires_grad = False
    
    def forward(self, images):
        """
        Args:
            images: [batch, 3, 224, 224] normalized images
        Returns:
            image_features: [batch, fusion_dim] projected image embedding
        """
        x = self.features(images)           # [batch, 2048, 7, 7]
        x = self.avgpool(x)                 # [batch, 2048, 1, 1]
        x = torch.flatten(x, 1)            # [batch, 2048]
        image_features = self.projection(x)  # [batch, fusion_dim]
        
        return image_features

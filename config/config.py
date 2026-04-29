"""
Configuration module for the Multimodal Meme Sentiment Analysis system.
All hyperparameters, paths, and settings are centralized here.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Config:
    """
    Central configuration class for the entire pipeline.
    Modify values here to control all aspects of training, evaluation, and inference.
    """
    
    # ============================================================
    # PROJECT PATHS
    # ============================================================
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = field(default=None)
    raw_data_dir: str = field(default=None)
    processed_data_dir: str = field(default=None)
    checkpoints_dir: str = field(default=None)
    results_dir: str = field(default=None)
    logs_dir: str = field(default=None)
    
    def __post_init__(self):
        """Initialize derived paths after dataclass creation."""
        if self.data_dir is None:
            self.data_dir = os.path.join(self.project_root, "data")
        if self.raw_data_dir is None:
            self.raw_data_dir = os.path.join(self.data_dir, "raw")
        if self.processed_data_dir is None:
            self.processed_data_dir = os.path.join(self.data_dir, "processed")
        if self.checkpoints_dir is None:
            self.checkpoints_dir = os.path.join(self.project_root, "checkpoints")
        if self.results_dir is None:
            self.results_dir = os.path.join(self.project_root, "results")
        if self.logs_dir is None:
            self.logs_dir = os.path.join(self.project_root, "logs")
        
        # Create directories
        for d in [self.data_dir, self.raw_data_dir, self.processed_data_dir,
                  self.checkpoints_dir, self.results_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)
    
    # ============================================================
    # DATASET SETTINGS
    # ============================================================
    kaggle_dataset: str = "williamscott701/memotion-dataset-7k"
    dataset_name: str = "memotion"
    
    # Train / Validation / Test split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    
    # ============================================================
    # MODEL ARCHITECTURE
    # ============================================================
    # Text encoder
    text_model_name: str = "xlm-roberta-base"
    text_embed_dim: int = 768
    max_text_length: int = 128
    text_freeze_layers: int = 8  # Freeze first N transformer layers (out of 12)
    
    # Image encoder
    image_model_name: str = "resnet50"
    image_embed_dim: int = 2048
    image_size: int = 224
    image_freeze_blocks: int = 6  # Freeze first N residual blocks (out of 8 total layers)
    
    # Fusion
    fusion_type: str = "cross_attention"  # Options: "cross_attention", "concat"
    fusion_dim: int = 512
    num_attention_heads: int = 4
    fusion_dropout: float = 0.3
    
    # Task heads (number of classes)
    num_sentiment_classes: int = 3   # positive, negative, neutral
    num_humor_classes: int = 4       # not_funny, funny, very_funny, hilarious
    num_sarcasm_classes: int = 2     # not_sarcastic, sarcastic (binary)
    
    # ============================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================
    batch_size: int = 16
    num_epochs: int = 20
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1  # Fraction of total steps for LR warmup
    max_grad_norm: float = 1.0  # Gradient clipping
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_f1_macro"  # Metric to monitor
    
    # Mixed precision training (AMP)
    use_amp: bool = True
    
    # ============================================================
    # LOSS WEIGHTS (for multi-task learning)
    # ============================================================
    # Relative importance of each task in the total loss
    sentiment_loss_weight: float = 1.0
    humor_loss_weight: float = 0.8
    sarcasm_loss_weight: float = 0.6
    
    # Whether to use learnable task weights (uncertainty-based)
    use_learnable_task_weights: bool = True
    
    # ============================================================
    # DATA AUGMENTATION
    # ============================================================
    use_image_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # ============================================================
    # OCR SETTINGS
    # ============================================================
    ocr_languages: list = field(default_factory=lambda: ["en", "hi"])
    ocr_confidence_threshold: float = 0.3
    use_dataset_text: bool = True  # Prefer dataset-provided text over OCR
    
    # ============================================================
    # DEVICE SETTINGS
    # ============================================================
    @property
    def device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    @property
    def num_workers(self) -> int:
        """Number of DataLoader workers (0 on Windows for compatibility)."""
        return 0 if os.name == 'nt' else 4
    
    # ============================================================
    # LABEL MAPPINGS
    # ============================================================
    @property
    def sentiment_labels(self) -> Dict[int, str]:
        return {0: "negative", 1: "neutral", 2: "positive"}
    
    @property
    def humor_labels(self) -> Dict[int, str]:
        return {0: "not_funny", 1: "funny", 2: "very_funny", 3: "hilarious"}
    
    @property
    def sarcasm_labels(self) -> Dict[int, str]:
        return {0: "not_sarcastic", 1: "sarcastic"}
    
    # ============================================================
    # EXPERIMENT PHASE CONFIGS
    # ============================================================
    def get_phase_config(self, phase: int) -> dict:
        """
        Returns phase-specific overrides for experimentation.
        
        Phase 1: Overfit (no proper split, no regularization)
        Phase 2: Improved (proper split, preprocessing, multi-task)
        Phase 3: Final (full pipeline, all optimizations)
        """
        phases = {
            1: {
                "description": "Phase 1: Demonstrate Overfitting",
                "train_ratio": 1.0,  # Use all data for training
                "val_ratio": 0.0,
                "test_ratio": 0.0,
                "num_epochs": 30,
                "use_image_augmentation": False,
                "early_stopping_patience": 999,  # Disabled
                "fusion_dropout": 0.0,
                "weight_decay": 0.0,
                "use_learnable_task_weights": False,
            },
            2: {
                "description": "Phase 2: Improved Pipeline",
                "train_ratio": 0.80,
                "val_ratio": 0.20,
                "test_ratio": 0.0,
                "num_epochs": 20,
                "use_image_augmentation": True,
                "early_stopping_patience": 5,
                "fusion_dropout": 0.2,
                "use_learnable_task_weights": True,
            },
            3: {
                "description": "Phase 3: Final Proper Evaluation",
                "train_ratio": 0.70,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "num_epochs": 20,
                "use_image_augmentation": True,
                "early_stopping_patience": 5,
                "fusion_dropout": 0.3,
                "use_learnable_task_weights": True,
            },
        }
        return phases.get(phase, phases[3])
    
    def __repr__(self):
        return (
            f"Config(\n"
            f"  device={self.device},\n"
            f"  text_model={self.text_model_name},\n"
            f"  image_model={self.image_model_name},\n"
            f"  fusion={self.fusion_type},\n"
            f"  batch_size={self.batch_size},\n"
            f"  lr={self.learning_rate},\n"
            f"  epochs={self.num_epochs}\n"
            f")"
        )

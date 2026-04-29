"""
Phase 1: Demonstrate Overfitting

This experiment trains the model WITHOUT a proper train/val/test split.
All data is used for training with no regularization — the model will
memorize the training data and show inflated metrics.

Purpose: Show WHY proper experimental methodology matters.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from data.preprocess import preprocess_dataset
from data.dataset import MemeDataset
from models.multimodal_model import MultimodalMemeAnalyzer
from training.trainer import Trainer
from utils.visualization import plot_training_curves


def run_phase1():
    """
    Phase 1: Overfit the model to demonstrate poor methodology.
    
    Key problems demonstrated:
    1. No validation/test split → can't measure generalization
    2. No regularization → model memorizes training data
    3. Training metrics look great but are meaningless
    """
    print("=" * 60)
    print("PHASE 1: OVERFITTING EXPERIMENT")
    print("=" * 60)
    print("\n⚠ This phase intentionally uses BAD practices to show why")
    print("  proper train/val/test splits are essential.\n")
    
    config = Config()
    phase_config = config.get_phase_config(1)
    
    # Apply phase-specific overrides
    for key, value in phase_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Preprocess with NO split (all data for training)
    train_df, _, _ = preprocess_dataset(
        config.data_dir,
        train_ratio=1.0, val_ratio=0.0, test_ratio=0.0,
        random_seed=config.random_seed,
        save_processed=False,
    )
    
    # Create dataset (NO augmentation)
    images_dir = os.path.join(config.data_dir, "raw", "images")
    train_dataset = MemeDataset(
        train_df, images_dir,
        tokenizer_name=config.text_model_name,
        max_length=config.max_text_length,
        image_size=config.image_size,
        augment=False,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        pin_memory=True,
    )
    
    # Create model (no dropout)
    model = MultimodalMemeAnalyzer(config)
    params = model.count_parameters()
    print(f"\nModel: {params['total_M']} params ({params['trainable_M']} trainable)")
    
    # Train WITHOUT validation
    trainer = Trainer(model, config, train_loader, val_loader=None)
    history = trainer.train()
    
    # Save results
    results_dir = os.path.join(config.results_dir, "phase1")
    plot_training_curves(history, results_dir)
    
    print("\n" + "=" * 60)
    print("PHASE 1 RESULTS:")
    print("=" * 60)
    print("Notice: Training accuracy is very high, but we have")
    print("NO validation metrics — we can't know if the model")
    print("actually learned anything useful or just memorized!")
    print("=" * 60)
    
    return history


if __name__ == "__main__":
    run_phase1()

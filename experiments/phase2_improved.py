"""
Phase 2: Improved Pipeline

Adds proper train/val split, Hinglish preprocessing, class balancing,
and multi-task learning. Shows clear improvement over Phase 1.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from data.preprocess import preprocess_dataset
from data.dataset import MemeDataset
from data.balancer import compute_class_weights, create_balanced_sampler
from models.multimodal_model import MultimodalMemeAnalyzer
from training.trainer import Trainer
from training.losses import MultiTaskLoss
from utils.hinglish import HinglishPreprocessor
from utils.visualization import plot_training_curves, plot_class_distribution


def run_phase2():
    """
    Phase 2: Improved methodology with validation.
    
    Improvements over Phase 1:
    1. Proper train/val split (80/20)
    2. Hinglish text preprocessing
    3. Class-weighted loss for imbalanced data
    4. Data augmentation
    5. Multi-task learning with learnable task weights
    """
    print("=" * 60)
    print("PHASE 2: IMPROVED PIPELINE")
    print("=" * 60)
    
    config = Config()
    phase_config = config.get_phase_config(2)
    for key, value in phase_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Preprocess with train/val split
    train_df, val_df, _ = preprocess_dataset(
        config.data_dir,
        train_ratio=0.80, val_ratio=0.20, test_ratio=0.0,
        random_seed=config.random_seed,
        save_processed=False,
    )
    
    # Apply Hinglish preprocessing
    print("\n🔤 Applying Hinglish preprocessing...")
    hinglish = HinglishPreprocessor()
    train_df['text_clean'] = train_df['text_clean'].apply(hinglish.preprocess)
    val_df['text_clean'] = val_df['text_clean'].apply(hinglish.preprocess)
    
    # Plot class distribution
    results_dir = os.path.join(config.results_dir, "phase2")
    plot_class_distribution(train_df, results_dir)
    
    # Compute class weights for balanced loss
    sentiment_weights = compute_class_weights(train_df['sentiment'].tolist(), config.num_sentiment_classes)
    humor_weights = compute_class_weights(train_df['humor'].tolist(), config.num_humor_classes)
    sarcasm_weights = compute_class_weights(train_df['sarcasm'].tolist(), config.num_sarcasm_classes)
    
    print(f"\n⚖️ Class weights:")
    print(f"  Sentiment: {sentiment_weights.tolist()}")
    print(f"  Humor: {humor_weights.tolist()}")
    print(f"  Sarcasm: {sarcasm_weights.tolist()}")
    
    # Create datasets
    images_dir = os.path.join(config.data_dir, "raw", "images")
    train_dataset = MemeDataset(train_df, images_dir, augment=True,
                                tokenizer_name=config.text_model_name,
                                max_length=config.max_text_length, image_size=config.image_size)
    val_dataset = MemeDataset(val_df, images_dir, augment=False,
                              tokenizer_name=config.text_model_name,
                              max_length=config.max_text_length, image_size=config.image_size)
    
    # Balanced sampler for training
    sampler = create_balanced_sampler(train_df['sentiment'].tolist())
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              sampler=sampler, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers, pin_memory=True)
    
    # Create model
    model = MultimodalMemeAnalyzer(config)
    
    # Train with validation
    trainer = Trainer(model, config, train_loader, val_loader)
    # Set class weights on loss
    trainer.criterion = MultiTaskLoss(config, sentiment_weights, humor_weights, sarcasm_weights).to(config.device)
    
    history = trainer.train()
    
    # Save results
    plot_training_curves(history, results_dir)
    
    print("\n" + "=" * 60)
    print("PHASE 2 RESULTS:")
    print("=" * 60)
    print("Now we can see the gap between train and val metrics.")
    print("This tells us how well the model generalizes!")
    print("=" * 60)
    
    return history


if __name__ == "__main__":
    run_phase2()

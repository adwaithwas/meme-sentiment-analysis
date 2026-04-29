"""
Phase 3: Final Proper Training & Evaluation

Full pipeline with:
- Proper 70/15/15 train/val/test split
- All preprocessing + class balancing
- Cross-attention fusion
- Complete evaluation on held-out test set
- All visualizations for the final report
"""

import os
import sys
import json
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
from training.metrics import compute_metrics, print_metrics, get_classification_report
from utils.hinglish import HinglishPreprocessor
from utils.visualization import (
    plot_training_curves, plot_confusion_matrices,
    plot_class_distribution
)


def run_phase3():
    """
    Phase 3: Final proper training and evaluation.
    
    This is the definitive experiment with:
    1. Stratified 70/15/15 split
    2. Full preprocessing + Hinglish normalization
    3. Class-weighted loss + balanced sampling
    4. Cross-attention fusion + multi-task learning
    5. Early stopping on validation F1
    6. Complete evaluation on held-out TEST set
    7. All visualizations for the report
    """
    print("=" * 60)
    print("PHASE 3: FINAL PROPER EVALUATION")
    print("=" * 60)
    
    config = Config()
    phase_config = config.get_phase_config(3)
    for key, value in phase_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Full preprocessing with 3-way split
    train_df, val_df, test_df = preprocess_dataset(
        config.data_dir,
        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
        random_seed=config.random_seed,
    )
    
    # Hinglish preprocessing
    print("\n🔤 Applying Hinglish preprocessing...")
    hinglish = HinglishPreprocessor()
    for df in [train_df, val_df, test_df]:
        df['text_clean'] = df['text_clean'].apply(hinglish.preprocess)
    
    results_dir = os.path.join(config.results_dir, "phase3")
    plot_class_distribution(train_df, results_dir)
    
    # Class weights
    sentiment_weights = compute_class_weights(train_df['sentiment'].tolist(), config.num_sentiment_classes)
    humor_weights = compute_class_weights(train_df['humor'].tolist(), config.num_humor_classes)
    sarcasm_weights = compute_class_weights(train_df['sarcasm'].tolist(), config.num_sarcasm_classes)
    
    # Datasets
    images_dir = os.path.join(config.data_dir, "raw", "images")
    train_dataset = MemeDataset(train_df, images_dir, augment=True,
                                tokenizer_name=config.text_model_name,
                                max_length=config.max_text_length, image_size=config.image_size)
    val_dataset = MemeDataset(val_df, images_dir, augment=False,
                              tokenizer_name=config.text_model_name,
                              max_length=config.max_text_length, image_size=config.image_size)
    test_dataset = MemeDataset(test_df, images_dir, augment=False,
                               tokenizer_name=config.text_model_name,
                               max_length=config.max_text_length, image_size=config.image_size)
    
    # DataLoaders
    sampler = create_balanced_sampler(train_df['sentiment'].tolist())
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              sampler=sampler, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=config.num_workers, pin_memory=True)
    
    # Model
    model = MultimodalMemeAnalyzer(config)
    params = model.count_parameters()
    print(f"\nModel: {params['total_M']} params ({params['trainable_M']} trainable)")
    
    # Train
    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.criterion = MultiTaskLoss(config, sentiment_weights, humor_weights, sarcasm_weights).to(config.device)
    history = trainer.train()
    
    # ---- FINAL EVALUATION ON TEST SET ----
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 60)
    
    # Load best model
    best_path = os.path.join(config.checkpoints_dir, "best_model.pt")
    if os.path.exists(best_path):
        trainer.load_checkpoint(best_path)
    
    # Evaluate on test set
    model.eval()
    all_preds = {'sentiment': [], 'humor': [], 'sarcasm': []}
    all_labels = {'sentiment': [], 'humor': [], 'sarcasm': []}
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(config.device)
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            
            outputs = model(input_ids, attention_mask, images)
            
            for task in ['sentiment', 'humor', 'sarcasm']:
                preds = outputs[f'{task}_logits'].argmax(dim=1).cpu().tolist()
                all_preds[task].extend(preds)
                all_labels[task].extend(batch[task].tolist())
    
    test_metrics = compute_metrics(all_preds, all_labels)
    
    print("\n📊 Test Set Results:")
    print_metrics(test_metrics, prefix="  Test")
    print(f"\n  Average F1 (macro): {test_metrics['avg_f1']:.4f}")
    
    # Classification reports
    label_maps = {
        'sentiment': config.sentiment_labels,
        'humor': config.humor_labels,
        'sarcasm': config.sarcasm_labels,
    }
    
    print("\n" + "-" * 40)
    for task in ['sentiment', 'humor', 'sarcasm']:
        names = list(label_maps[task].values())
        report = get_classification_report(all_preds[task], all_labels[task], names)
        print(f"\n{task.upper()} Classification Report:\n{report}")
    
    # Visualizations
    plot_training_curves(history, results_dir)
    plot_confusion_matrices(all_preds, all_labels, label_maps, results_dir)
    
    # Save metrics to JSON
    metrics_json = {}
    for task in ['sentiment', 'humor', 'sarcasm']:
        m = test_metrics[task]
        metrics_json[task] = {
            'accuracy': round(m['accuracy'], 4),
            'precision': round(m['precision'], 4),
            'recall': round(m['recall'], 4),
            'f1': round(m['f1'], 4),
        }
    metrics_json['avg_f1'] = round(test_metrics['avg_f1'], 4)
    
    metrics_path = os.path.join(results_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"\n✓ Saved test metrics to {metrics_path}")
    
    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETE!")
    print(f"Results saved to: {results_dir}")
    print("=" * 60)
    
    return history, test_metrics


if __name__ == "__main__":
    run_phase3()

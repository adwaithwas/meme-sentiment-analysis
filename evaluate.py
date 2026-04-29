import os
import sys
import torch
import traceback
import json

if sys.stdout.encoding != 'utf-8' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from data.preprocess import preprocess_dataset
from data.dataset import MemeDataset
from torch.utils.data import DataLoader
from models.multimodal_model import MultimodalMemeAnalyzer
from training.metrics import compute_metrics
from utils.visualization import plot_confusion_matrices

def evaluate_only():
    print("=" * 60)
    print("RUNNING FINAL EVALUATION (NO RETRAINING)")
    print("=" * 60)
    
    config = Config()
    
    train_df, val_df, test_df = preprocess_dataset(
        config.data_dir,
        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
        random_seed=config.random_seed,
    )
    
    images_dir = os.path.join(config.data_dir, "raw", "images")
    test_dataset = MemeDataset(test_df, images_dir, augment=False,
                               tokenizer_name=config.text_model_name,
                               max_length=config.max_text_length, image_size=config.image_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=config.num_workers, pin_memory=True)
                             
    model = MultimodalMemeAnalyzer(config)
    model.to(config.device)
    
    best_path = os.path.join(config.checkpoints_dir, "best_model.pt")
    if not os.path.exists(best_path):
        print(f"Error: {best_path} not found!")
        return
        
    print(f"Loading checkpoint from {best_path}...")
    checkpoint = torch.load(best_path, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = {'sentiment': [], 'humor': [], 'sarcasm': []}
    all_labels = {'sentiment': [], 'humor': [], 'sarcasm': []}
    
    print("Evaluating on test set...")
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
                
    print("Computing metrics...")
    test_metrics = compute_metrics(all_preds, all_labels)
    
    results_dir = os.path.join(config.results_dir, "phase3")
    os.makedirs(results_dir, exist_ok=True)
    
    label_maps = {
        'sentiment': config.sentiment_labels,
        'humor': config.humor_labels,
        'sarcasm': config.sarcasm_labels,
    }
    plot_confusion_matrices(all_preds, all_labels, label_maps, results_dir)
    
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
        
    print(f"\n✅ Success! Evaluation complete.")
    print(f"Results and plots saved to: {results_dir}")

if __name__ == '__main__':
    try:
        evaluate_only()
    except Exception as e:
        traceback.print_exc()

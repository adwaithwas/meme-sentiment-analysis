"""
Visualization utilities for training curves, confusion matrices, and predictions.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Set global style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_curves(history, save_dir):
    """Plot loss and F1 curves over epochs."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    if history.get('val_loss'):
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    
    # F1 curve
    axes[1].plot(history['train_f1'], label='Train F1', linewidth=2)
    if history.get('val_f1'):
        axes[1].plot(history['val_f1'], label='Val F1', linewidth=2, linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Macro F1')
    axes[1].set_title('Training & Validation F1 Score')
    axes[1].legend()
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved training curves: {path}")


def plot_confusion_matrices(all_preds, all_labels, label_maps, save_dir):
    """Plot confusion matrix heatmaps for each task."""
    os.makedirs(save_dir, exist_ok=True)
    
    tasks = ['sentiment', 'humor', 'sarcasm']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, task in enumerate(tasks):
        preds = all_preds[task]
        labels = all_labels[task]
        label_names = list(label_maps[task].values())
        
        cm = confusion_matrix(labels, preds)
        
        # Normalize by row (true labels)
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names,
                    ax=axes[idx], cbar=True)
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
        axes[idx].set_title(f'{task.capitalize()} Confusion Matrix')
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'confusion_matrices.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved confusion matrices: {path}")


def plot_class_distribution(df, save_dir):
    """Plot class distribution bar charts for each task."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    task_info = {
        'sentiment': {0: 'Negative', 1: 'Neutral', 2: 'Positive'},
        'humor': {0: 'Not Funny', 1: 'Funny', 2: 'Very Funny', 3: 'Hilarious'},
        'sarcasm': {0: 'Not Sarcastic', 1: 'Sarcastic'},
    }
    
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    
    for idx, (task, label_map) in enumerate(task_info.items()):
        if task not in df.columns:
            continue
        counts = df[task].value_counts().sort_index()
        names = [label_map.get(i, str(i)) for i in counts.index]
        
        bars = axes[idx].bar(names, counts.values, color=colors[:len(names)], edgecolor='white')
        axes[idx].set_title(f'{task.capitalize()} Distribution')
        axes[idx].set_ylabel('Count')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts.values):
            axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                          str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'class_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved class distribution: {path}")


def plot_metrics_comparison(phase_results, save_dir):
    """Compare metrics across experiment phases (bar chart)."""
    os.makedirs(save_dir, exist_ok=True)
    
    tasks = ['sentiment', 'humor', 'sarcasm']
    phases = list(phase_results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(phases))
    width = 0.2
    
    for idx, task in enumerate(tasks):
        metrics_list = ['accuracy', 'precision', 'recall', 'f1']
        for j, metric in enumerate(metrics_list):
            values = [phase_results[p].get(task, {}).get(metric, 0) for p in phases]
            axes[idx].bar(x + j * width, values, width, label=metric.capitalize())
        
        axes[idx].set_xlabel('Phase')
        axes[idx].set_ylabel('Score')
        axes[idx].set_title(f'{task.capitalize()} Metrics')
        axes[idx].set_xticks(x + width * 1.5)
        axes[idx].set_xticklabels(phases)
        axes[idx].legend(fontsize=8)
        axes[idx].set_ylim(0, 1.0)
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'phase_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved phase comparison: {path}")

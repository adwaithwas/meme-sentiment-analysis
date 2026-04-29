"""
Class balancing utilities for handling imbalanced meme datasets.
Provides weighted sampling and class weight computation.
"""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from collections import Counter


def compute_class_weights(labels, num_classes=None):
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.
    Higher weight = rarer class gets more importance.
    
    Args:
        labels: list/array of integer labels
        num_classes: total number of classes (auto-detected if None)
    Returns:
        torch.FloatTensor of weights per class
    """
    counter = Counter(labels)
    if num_classes is None:
        num_classes = max(counter.keys()) + 1
    
    total = sum(counter.values())
    weights = []
    for i in range(num_classes):
        count = counter.get(i, 1)  # Avoid division by zero
        weights.append(total / (num_classes * count))
    
    weights = torch.FloatTensor(weights)
    # Normalize so weights sum to num_classes
    weights = weights / weights.sum() * num_classes
    return weights


def create_balanced_sampler(labels):
    """
    Create a WeightedRandomSampler for balanced training batches.
    Each sample gets a weight inversely proportional to its class frequency.
    
    Args:
        labels: list/array of integer labels for each sample
    Returns:
        WeightedRandomSampler instance
    """
    counter = Counter(labels)
    total = len(labels)
    
    # Weight per class
    class_weights = {cls: total / count for cls, count in counter.items()}
    
    # Weight per sample
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def print_class_distribution(labels, label_names=None, title="Distribution"):
    """Print a formatted class distribution summary."""
    counter = Counter(labels)
    total = sum(counter.values())
    print(f"\n  {title}:")
    for cls in sorted(counter.keys()):
        count = counter[cls]
        pct = count / total * 100
        name = label_names[cls] if label_names else str(cls)
        bar = "█" * int(pct / 2)
        print(f"    {name:>15s}: {count:5d} ({pct:5.1f}%) {bar}")

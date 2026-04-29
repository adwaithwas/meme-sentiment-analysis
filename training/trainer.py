"""
Training loop with validation, checkpointing, and mixed-precision support.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from training.metrics import compute_metrics, print_metrics
from training.losses import MultiTaskLoss


class Trainer:
    """
    Handles the full training lifecycle:
    - Training loop with mixed precision (AMP)
    - Validation after each epoch
    - Early stopping based on validation F1
    - Model checkpointing (saves best model)
    - Learning rate scheduling (cosine with warmup)
    - Gradient clipping for stability
    """
    
    def __init__(self, model, config, train_loader, val_loader=None):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device
        
        # Multi-task loss
        self.criterion = MultiTaskLoss(config).to(self.device)
        
        # Optimizer: AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler: cosine with warmup
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=config.warmup_ratio,
            anneal_strategy='cos',
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_amp and self.device.type == 'cuda' else None
        
        # Tracking
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    
    def train_epoch(self, epoch):
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        all_preds = {'sentiment': [], 'humor': [], 'sarcasm': []}
        all_labels = {'sentiment': [], 'humor': [], 'sarcasm': []}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        
        for batch in pbar:
            # Move to device
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = {
                'sentiment': batch['sentiment'].to(self.device),
                'humor': batch['humor'].to(self.device),
                'sarcasm': batch['sarcasm'].to(self.device),
            }
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    outputs = self.model(input_ids, attention_mask, images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(input_ids, attention_mask, images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Collect predictions
            for task in ['sentiment', 'humor', 'sarcasm']:
                preds = outputs[f'{task}_logits'].argmax(dim=1).cpu().tolist()
                all_preds[task].extend(preds)
                all_labels[task].extend(labels[task].cpu().tolist())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'})
        
        avg_loss = total_loss / len(self.train_loader)
        metrics = compute_metrics(all_preds, all_labels)
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def validate(self, epoch):
        """Run validation."""
        if self.val_loader is None:
            return None, None
        
        self.model.eval()
        total_loss = 0
        all_preds = {'sentiment': [], 'humor': [], 'sarcasm': []}
        all_labels = {'sentiment': [], 'humor': [], 'sarcasm': []}
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = {
                'sentiment': batch['sentiment'].to(self.device),
                'humor': batch['humor'].to(self.device),
                'sarcasm': batch['sarcasm'].to(self.device),
            }
            
            if self.scaler:
                with autocast():
                    outputs = self.model(input_ids, attention_mask, images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(input_ids, attention_mask, images)
                loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            
            for task in ['sentiment', 'humor', 'sarcasm']:
                preds = outputs[f'{task}_logits'].argmax(dim=1).cpu().tolist()
                all_preds[task].extend(preds)
                all_labels[task].extend(labels[task].cpu().tolist())
        
        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_metrics(all_preds, all_labels)
        
        return avg_loss, metrics
    
    def train(self):
        """Full training loop with early stopping and checkpointing."""
        print(f"\n🚀 Starting training on {self.device}")
        print(f"   Epochs: {self.config.num_epochs} | Batch size: {self.config.batch_size}")
        print(f"   LR: {self.config.learning_rate} | Early stopping: {self.config.early_stopping_patience}")
        params = self.model.count_parameters()
        print(f"   Parameters: {params['total_M']} total, {params['trainable_M']} trainable\n")
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_metrics['avg_f1'])
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            
            elapsed = time.time() - start_time
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs} ({elapsed:.0f}s)")
            print(f"  Train Loss: {train_loss:.4f} | F1: {train_metrics['avg_f1']:.4f}")
            print_metrics(train_metrics, prefix="  Train")
            
            if val_metrics:
                self.history['val_loss'].append(val_loss)
                self.history['val_f1'].append(val_metrics['avg_f1'])
                
                print(f"  Val   Loss: {val_loss:.4f} | F1: {val_metrics['avg_f1']:.4f}")
                print_metrics(val_metrics, prefix="  Val  ")
                
                # Check for improvement
                if val_metrics['avg_f1'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['avg_f1']
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, val_metrics, is_best=True)
                    print(f"  ✓ New best model! F1: {self.best_val_f1:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"  No improvement ({self.patience_counter}/{self.config.early_stopping_patience})")
                    
                    if self.patience_counter >= self.config.early_stopping_patience:
                        print(f"\n⚡ Early stopping at epoch {epoch+1}")
                        break
            else:
                # No validation — save every 5 epochs
                if (epoch + 1) % 5 == 0:
                    self._save_checkpoint(epoch, train_metrics)
        
        print(f"\n✅ Training complete! Best val F1: {self.best_val_f1:.4f}")
        return self.history
    
    def _save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint."""
        os.makedirs(self.config.checkpoints_dir, exist_ok=True)
        
        filename = "best_model.pt" if is_best else f"checkpoint_epoch{epoch+1}.pt"
        path = os.path.join(self.config.checkpoints_dir, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }, path)
    
    def load_checkpoint(self, path):
        """Load a saved checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Loaded checkpoint from {path} (epoch {checkpoint['epoch']+1})")
        return checkpoint

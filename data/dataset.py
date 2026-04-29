"""
PyTorch Dataset class for multimodal meme data.
Loads image + text + multi-label targets for training.
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import XLMRobertaTokenizer


class MemeDataset(Dataset):
    """
    Multimodal dataset: each sample has an image, tokenized text, and multi-task labels.
    
    Returns dict with keys:
        image: tensor [3, 224, 224]
        input_ids: tensor [max_length]
        attention_mask: tensor [max_length]
        sentiment: int (0-2)
        humor: int (0-3)
        sarcasm: int (0-1)
    """
    
    def __init__(self, df, images_dir, tokenizer_name='xlm-roberta-base',
                 max_length=128, image_size=224, augment=False):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.max_length = max_length
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_name)
        
        # Image transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- Load Image ---
        img_path = os.path.join(self.images_dir, row['image_name'])
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception:
            # Fallback: black image if file is corrupted
            image = torch.zeros(3, 224, 224)
        
        # --- Tokenize Text ---
        text = row.get('text_clean', row.get('text', ''))
        if not isinstance(text, str):
            text = ""
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'sentiment': torch.tensor(int(row['sentiment']), dtype=torch.long),
            'humor': torch.tensor(int(row['humor']), dtype=torch.long),
            'sarcasm': torch.tensor(int(row['sarcasm']), dtype=torch.long),
        }

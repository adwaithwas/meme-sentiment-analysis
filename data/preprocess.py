"""
Text preprocessing and label mapping for the Memotion dataset.
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

SENTIMENT_MAP = {"negative": 0, "neutral": 1, "positive": 2, "very_negative": 0, "very_positive": 2}
HUMOR_MAP = {"not_funny": 0, "funny": 1, "very_funny": 2, "hilarious": 3, 0: 0, 1: 1, 2: 2, 3: 3}
SARCASM_MAP = {"not_sarcastic": 0, "general": 1, "twisted_meaning": 1, "very_twisted": 1, 0: 0, 1: 1, 2: 1, 3: 1}
OFFENSIVE_MAP = {"not_offensive": 0, "slight": 1, "very_offensive": 2, "hateful_offensive": 3, 0: 0, 1: 1, 2: 2, 3: 3}

EMOJI_MAP = {
    '😂': ' laughing ', '😭': ' crying ', '❤️': ' love ', '🔥': ' fire hot ',
    '💀': ' dead funny ', '😍': ' love eyes ', '🤣': ' laughing hard ', '😎': ' cool ',
    '😡': ' angry ', '😢': ' sad ', '🙄': ' eyeroll sarcastic ', '👍': ' good ',
    '👎': ' bad ', '🤔': ' thinking ', '😏': ' smirk sarcastic ', '💔': ' heartbreak sad ',
    '😤': ' frustrated ', '🤡': ' clown foolish ', '💪': ' strong ', '🙏': ' please pray ',
}


def clean_text(text: str) -> str:
    """Clean meme text: remove URLs, mentions, convert emojis, normalize whitespace."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    for emoji_char, replacement in EMOJI_MAP.items():
        text = text.replace(emoji_char, replacement)
    try:
        import emoji
        text = emoji.demojize(text, delimiters=(" ", " "))
        text = re.sub(r':(\w+):', lambda m: m.group(1).replace('_', ' '), text)
    except ImportError:
        pass
    text = re.sub(r'[^\w\s.,!?\'"-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def parse_memotion_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and normalize labels from Memotion dataset CSV."""
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    print(f"  Columns found: {list(df.columns)}")
    
    col_map = {}
    for col in ['image_name', 'image', 'img', 'filename', 'file_name']:
        if col in df.columns:
            col_map[col] = 'image_name'; break
    for col in ['text_ocr', 'text', 'ocr_text', 'caption', 'text_corrected']:
        if col in df.columns and col not in col_map:
            col_map[col] = 'text'; break
    for col in ['overall_sentiment', 'sentiment', 'label', 'overall']:
        if col in df.columns:
            col_map[col] = 'sentiment'; break
    for col in ['humour', 'humor', 'humorous']:
        if col in df.columns:
            col_map[col] = 'humor'; break
    for col in ['sarcasm', 'sarcastic']:
        if col in df.columns:
            col_map[col] = 'sarcasm'; break
    for col in ['offensive', 'offensiveness']:
        if col in df.columns:
            col_map[col] = 'offensive'; break
    
    df = df.rename(columns=col_map)
    if 'image_name' not in df.columns:
        raise ValueError(f"Image column not found. Available: {list(df.columns)}")
    if 'sentiment' not in df.columns:
        raise ValueError(f"Sentiment column not found. Available: {list(df.columns)}")
    if 'text' not in df.columns:
        df['text'] = ""
    
    df['text'] = df['text'].fillna("").astype(str)
    
    # Encode sentiment
    if not pd.api.types.is_numeric_dtype(df['sentiment']):
        df['sentiment'] = df['sentiment'].astype(str).str.strip().str.lower().map(SENTIMENT_MAP)
    df['sentiment'] = df['sentiment'].fillna(1).astype(int)
    
    # Encode humor
    if 'humor' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['humor']):
            df['humor'] = df['humor'].astype(str).str.strip().str.lower().map(HUMOR_MAP)
        df['humor'] = df['humor'].fillna(0).astype(int)
    else:
        df['humor'] = 0
    
    # Encode sarcasm (binary)
    if 'sarcasm' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['sarcasm']):
            df['sarcasm'] = df['sarcasm'].astype(str).str.strip().str.lower().map(SARCASM_MAP)
        else:
            df['sarcasm'] = (df['sarcasm'] > 0).astype(int)
        df['sarcasm'] = df['sarcasm'].fillna(0).astype(int)
    else:
        df['sarcasm'] = 0
    
    # Encode offensive
    if 'offensive' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['offensive']):
            df['offensive'] = df['offensive'].astype(str).str.strip().str.lower().map(OFFENSIVE_MAP)
        df['offensive'] = df['offensive'].fillna(0).astype(int)
    else:
        df['offensive'] = 0
    
    df['text_clean'] = df['text'].apply(clean_text)
    df = df.dropna(subset=['image_name'])
    df['image_name'] = df['image_name'].astype(str).str.strip()
    
    return df[['image_name', 'text', 'text_clean', 'sentiment', 'humor', 'sarcasm', 'offensive']]


def create_splits(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42, stratify_col='sentiment'):
    """Create stratified train/val/test splits."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    if val_ratio == 0 and test_ratio == 0:
        print(f"  ⚠ No split — all {len(df)} samples for training")
        return df, None, None
    if test_ratio == 0:
        train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=random_seed, stratify=df[stratify_col])
        print(f"  Train: {len(train_df)} | Val: {len(val_df)}")
        return train_df, val_df, None
    
    train_val_df, test_df = train_test_split(df, test_size=test_ratio, random_state=random_seed, stratify=df[stratify_col])
    rel_val = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(train_val_df, test_size=rel_val, random_state=random_seed, stratify=train_val_df[stratify_col])
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def preprocess_dataset(data_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42, save_processed=True):
    """Full pipeline: load CSV → parse labels → clean text → split."""
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    labels_path = os.path.join(raw_dir, "labels.csv")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at {labels_path}. Run data/download.py first.")
    
    print("📊 Loading and preprocessing dataset...")
    df = pd.read_csv(labels_path, encoding='utf-8', on_bad_lines='skip')
    print(f"  Raw dataset: {len(df)} samples")
    
    df = parse_memotion_labels(df)
    print(f"  After cleaning: {len(df)} samples")
    
    # Filter missing images
    images_dir = os.path.join(raw_dir, "images")
    if os.path.exists(images_dir):
        existing = set(os.listdir(images_dir))
        before = len(df)
        df = df[df['image_name'].isin(existing)]
        if len(df) < before:
            print(f"  Filtered missing images: {before} → {len(df)}")
    
    # Print distribution
    print("\n📊 Label Distribution:")
    for col in ['sentiment', 'humor', 'sarcasm']:
        counts = df[col].value_counts().sort_index()
        print(f"\n  {col.upper()}:")
        for val, count in counts.items():
            print(f"    {val}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\n✂️ Creating splits ({train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%})...")
    train_df, val_df, test_df = create_splits(df, train_ratio, val_ratio, test_ratio, random_seed)
    
    if save_processed:
        train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
        if val_df is not None:
            val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
        if test_df is not None:
            test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
        print("  ✓ Saved processed CSVs")
    
    return train_df, val_df, test_df

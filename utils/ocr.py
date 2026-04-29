"""
EasyOCR wrapper for extracting text from meme images.
Supports English + Hindi for Hinglish memes.
"""

import os
from typing import List, Optional


# Lazy load EasyOCR (heavy import)
_reader = None


def _get_reader(languages=None):
    """Get or create EasyOCR reader (singleton pattern)."""
    global _reader
    if _reader is None:
        import easyocr
        langs = languages or ['en', 'hi']
        _reader = easyocr.Reader(langs, gpu=True)
        print(f"✓ EasyOCR initialized with languages: {langs}")
    return _reader


def extract_text_from_image(image_path: str, languages=None,
                            confidence_threshold: float = 0.3) -> str:
    """
    Extract text from a meme image using EasyOCR.
    
    Args:
        image_path: Path to the image file
        languages: List of language codes (default: ['en', 'hi'])
        confidence_threshold: Minimum confidence to include text
        
    Returns:
        Extracted text as a single string
    """
    if not os.path.exists(image_path):
        return ""
    
    try:
        reader = _get_reader(languages)
        results = reader.readtext(image_path)
        
        # Filter by confidence and join text
        texts = []
        for (bbox, text, confidence) in results:
            if confidence >= confidence_threshold:
                texts.append(text.strip())
        
        return " ".join(texts)
    
    except Exception as e:
        print(f"⚠ OCR failed for {image_path}: {e}")
        return ""


def batch_extract_text(image_paths: List[str], languages=None,
                       confidence_threshold: float = 0.3) -> List[str]:
    """Extract text from multiple images."""
    return [extract_text_from_image(p, languages, confidence_threshold)
            for p in image_paths]

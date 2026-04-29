"""
CLI inference: predict sentiment, humor, and sarcasm from a meme image.
Usage: python predict.py --image path/to/meme.jpg
"""

import argparse
import sys
import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import XLMRobertaTokenizer

if sys.stdout.encoding != 'utf-8' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from models.multimodal_model import MultimodalMemeAnalyzer
from utils.ocr import extract_text_from_image
from utils.hinglish import HinglishPreprocessor
from data.preprocess import clean_text


def predict_meme(image_path, model=None, config=None, text_override=None):
    """
    Run full inference on a single meme image.
    
    Pipeline: Image → OCR → Hinglish preprocessing → Model → Predictions
    """
    if config is None:
        config = Config()
    
    # Load model if not provided
    if model is None:
        model = MultimodalMemeAnalyzer(config)
        checkpoint_path = os.path.join(config.checkpoints_dir, "best_model.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded model from {checkpoint_path}")
        else:
            print("⚠ No trained model found! Using untrained model (random predictions).")
        model = model.to(config.device)
        model.eval()
    
    # --- Extract text from image ---
    if text_override:
        raw_text = text_override
    else:
        print("🔍 Extracting text from image...")
        raw_text = extract_text_from_image(image_path, config.ocr_languages)
    
    print(f"  Raw text: {raw_text}")
    
    # Clean and preprocess
    cleaned = clean_text(raw_text)
    hinglish = HinglishPreprocessor()
    processed = hinglish.preprocess(cleaned)
    print(f"  Processed: {processed}")
    
    # --- Prepare image ---
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(config.device)
    
    # --- Tokenize text ---
    tokenizer = XLMRobertaTokenizer.from_pretrained(config.text_model_name)
    encoding = tokenizer(processed, max_length=config.max_text_length,
                         padding='max_length', truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids'].to(config.device)
    attention_mask = encoding['attention_mask'].to(config.device)
    
    # --- Inference ---
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, image_tensor)
    
    # --- Format results ---
    results = {}
    
    # Sentiment
    sent_probs = torch.softmax(outputs['sentiment_logits'], dim=1).squeeze().cpu()
    sent_pred = sent_probs.argmax().item()
    results['sentiment'] = {
        'prediction': config.sentiment_labels[sent_pred],
        'confidence': f"{sent_probs[sent_pred]:.1%}",
        'probabilities': {config.sentiment_labels[i]: f"{sent_probs[i]:.1%}" for i in range(3)},
    }
    
    # Humor
    humor_probs = torch.softmax(outputs['humor_logits'], dim=1).squeeze().cpu()
    humor_pred = humor_probs.argmax().item()
    results['humor'] = {
        'prediction': config.humor_labels[humor_pred],
        'confidence': f"{humor_probs[humor_pred]:.1%}",
        'probabilities': {config.humor_labels[i]: f"{humor_probs[i]:.1%}" for i in range(4)},
    }
    
    # Sarcasm
    sarc_probs = torch.softmax(outputs['sarcasm_logits'], dim=1).squeeze().cpu()
    sarc_pred = sarc_probs.argmax().item()
    results['sarcasm'] = {
        'prediction': config.sarcasm_labels[sarc_pred],
        'confidence': f"{sarc_probs[sarc_pred]:.1%}",
        'probabilities': {config.sarcasm_labels[i]: f"{sarc_probs[i]:.1%}" for i in range(2)},
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict meme sentiment, humor, and sarcasm")
    parser.add_argument('--image', type=str, required=True, help='Path to meme image')
    parser.add_argument('--text', type=str, default=None, help='Override OCR with manual text')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)
    
    results = predict_meme(args.image, text_override=args.text)
    
    print("\n" + "=" * 50)
    print("🎭 MEME ANALYSIS RESULTS")
    print("=" * 50)
    
    for task, info in results.items():
        print(f"\n  {task.upper()}: {info['prediction']} ({info['confidence']})")
        for label, prob in info['probabilities'].items():
            bar = "█" * int(float(prob.strip('%')) / 5)
            print(f"    {label:>15s}: {prob:>6s} {bar}")


if __name__ == "__main__":
    main()

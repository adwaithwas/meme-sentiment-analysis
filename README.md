# Multimodal Meme Sentiment Analysis

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/adwaithwas/meme-sentiment-analysis)

A deep learning system for analyzing memes — predicting **sentiment**, **humor type**, and **sarcasm** using multimodal fusion of text (XLM-RoBERTa) and images (ResNet50) with cross-attention.

## Architecture

```
Image → ResNet50 → projection → ┐
                                 ├→ Cross-Attention Fusion → Task Heads
Text  → XLM-RoBERTa → projection → ┘
```

## Attributes
- Sentiment: (positive / negative / neutral)
- Humor: (not funny → hilarious)
- Sarcasm: (binary detection)

## Dataset

Uses the **Memotion Dataset 7K** (SemEval-2020) with:
- ~7000 meme images with annotations
- Hinglish (Hindi-English code-mixed) text support
- Class-balanced training via weighted sampling

### Setup
creating virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```
install dependencies
```
pip install -r requirements.txt
```

### Download Model Weights
Due to file size limits, the trained model weights are hosted on Hugging Face.
1. Download `best_model.pt` from [Hugging Face](https://huggingface.co/adwaithwas/meme-sentiment-analysis/resolve/main/best_model.pt).
2. Create a `checkpoints/` folder in the project root.
3. Place the file inside: `checkpoints/best_model.pt`.

### Download Dataset (Optional)
```bash
python train.py --download --phase 3
```

```bash
# Phase 1: Overfit demo (shows why proper splits matter)
python train.py --phase 1

# Phase 2: Improved pipeline (train/val split + balancing)
python train.py --phase 2

# Phase 3: Final proper evaluation (recommended)
python train.py --phase 3
```
```bash
python predict.py --image path/to/meme.jpg
```

### Web Demo
```bash
python webapp/app.py
# Open http://localhost:5000
```

## Project Structure

```
meme/
├── config/config.py           # All hyperparameters
├── data/                      # Data pipeline
│   ├── download.py            # Kaggle download
│   ├── preprocess.py          # Text cleaning + splits
│   ├── dataset.py             # PyTorch Dataset
│   └── balancer.py            # Class balancing
├── models/                    # Model architecture
│   ├── text_encoder.py        # XLM-RoBERTa
│   ├── image_encoder.py       # ResNet50
│   ├── fusion.py              # Cross-attention
│   └── multimodal_model.py    # Full model
├── training/                  # Training pipeline
│   ├── trainer.py             # Training loop
│   ├── losses.py              # Multi-task loss
│   └── metrics.py             # Evaluation metrics
├── utils/                     # Utilities
│   ├── ocr.py                 # EasyOCR
│   ├── hinglish.py            # Hinglish preprocessing
│   └── visualization.py       # Plots
├── experiments/               # 3 experiment phases
├── webapp/                    # Flask web demo
├── train.py                   # Training entry point
├── predict.py                 # CLI inference
└── start.bat                  # Windows launcher
```

## Experiment Phases

| Phase | Split | Augmentation | Balancing | Purpose |
|-------|-------|-------------|-----------|---------|
| 1 | None (100% train) | ✗ | ✗ | Show overfitting |
| 2 | 80/20 | ✓ | ✓ | Improved pipeline |
| 3 | 70/15/15 | ✓ | ✓ | Final evaluation |

## 🛠️ Tech Stack

- **PyTorch** — Deep learning framework
- **XLM-RoBERTa** — Multilingual text encoder (Hinglish support)
- **ResNet50** — Image feature extraction
- **EasyOCR** — Text extraction from memes
- **Flask** — Web demo
- **scikit-learn** — Metrics & evaluation



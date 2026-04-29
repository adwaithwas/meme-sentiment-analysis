"""
Main training entry point.
Usage: python train.py --phase 1|2|3
"""

import argparse
import sys
import os

if sys.stdout.encoding != 'utf-8' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Multimodal Meme Sentiment Analysis - Training")
    parser.add_argument('--phase', type=int, default=3, choices=[1, 2, 3],
                        help='Experiment phase: 1=overfit, 2=improved, 3=final (default: 3)')
    parser.add_argument('--download', action='store_true',
                        help='Download dataset from Kaggle before training')
    args = parser.parse_args()
    
    # Download dataset if requested
    if args.download:
        from data.download import download_memotion_dataset, verify_dataset
        from config.config import Config
        config = Config()
        success = download_memotion_dataset(config.data_dir, config.kaggle_dataset)
        if not success:
            print("\n❌ Dataset download failed. Please download manually.")
            sys.exit(1)
        if not verify_dataset(config.data_dir):
            sys.exit(1)
    
    # Run selected phase
    print(f"\n🔬 Running Phase {args.phase}...\n")
    
    if args.phase == 1:
        from experiments.phase1_overfit import run_phase1
        run_phase1()
    elif args.phase == 2:
        from experiments.phase2_improved import run_phase2
        run_phase2()
    elif args.phase == 3:
        from experiments.phase3_final import run_phase3
        run_phase3()


if __name__ == "__main__":
    main()

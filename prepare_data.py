"""
Data Preparation Pipeline

Runs both:
1. Dataset preprocessing (data/raw/dataset.jsonl -> data/processed/dataset.jsonl)
2. Stratified splitting (train, validation, test)

Usage:
    python prepare_data.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocessing.preprocess_dataset import DatasetPreprocessor
from src.dataset.splitter import DatasetSplitter
from configs.paths import PROCESSED_DATASET_PATH, PROCESSED_DATA_DIR


def main():
    print("=" * 60)
    print("STEP 1: PREPROCESSING RAW DATASET")
    print("=" * 60)
    preprocessor = DatasetPreprocessor()
    preprocessor.process()

    print("\n" + "=" * 60)
    print("STEP 2: SPLITTING DATASET (TRAIN / VAL / TEST)")
    print("=" * 60)
    splitter = DatasetSplitter()
    splitter.execute(
        dataset_path=str(PROCESSED_DATASET_PATH),
        output_directory=str(PROCESSED_DATA_DIR)
    )

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()

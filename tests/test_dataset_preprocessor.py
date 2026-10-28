import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.preprocess_dataset import DatasetPreprocessor

processor = DatasetPreprocessor()

processor.process()

print()

print("Dataset preprocessing completed successfully.")
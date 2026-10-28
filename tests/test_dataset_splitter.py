import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset.splitter import DatasetSplitter

splitter = DatasetSplitter()

splitter.execute(

    dataset_path="data/processed/dataset.jsonl",

    output_directory="data/processed"

)
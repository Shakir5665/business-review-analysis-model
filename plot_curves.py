"""
Root shortcut to plot training curves from saved history.json.

Usage:
    python plot_curves.py
    python plot_curves.py --history outputs/reports/history.json
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.visualization.plot_curves import plot_training_curves

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot training curves from history.json")
    parser.add_argument("--history", type=str, default=None, help="Path to history.json (default: auto-detected)")
    parser.add_argument("--save-dir", type=str, default="outputs/plots", help="Directory to save plot image")
    parser.add_argument("--show", action="store_true", help="Display plot interactively")
    args = parser.parse_args()

    plot_training_curves(history=args.history, save_dir=args.save_dir, show=args.show)

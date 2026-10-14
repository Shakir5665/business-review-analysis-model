"""
Training Curves Visualizer

Standalone script to plot loss and accuracy curves from training history JSON,
including markers for the best model checkpoint and early stopping trigger.

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

import json
from pathlib import Path
import shutil
from typing import Optional, Union, Dict, Any

import matplotlib.pyplot as plt

from configs.training_config import GDRIVE_REPORTS_DIR, GDRIVE_CHECKPOINT_DIR, GDRIVE_PLOTS_DIR


def load_history(history_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load history dictionary from the specified or standard search paths.
    """
    candidates = []
    if history_path is not None:
        candidates.append(Path(history_path))

    candidates.extend([
        Path("history.json"),
        Path("outputs/reports/history.json"),
        Path("checkpoints/history.json"),
        Path(f"{GDRIVE_REPORTS_DIR}/history.json"),
        Path(f"{GDRIVE_CHECKPOINT_DIR}/history.json")
    ])

    for p in candidates:
        if p.exists():
            with open(p, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and "epochs" in data:
                    print(f"Loaded training history from: {p}")
                    return data

    raise FileNotFoundError(
        f"Could not find history.json in any expected locations:\n" +
        "\n".join(f" - {c}" for c in candidates)
    )


def plot_training_curves(
    history: Optional[Union[Dict[str, Any], str, Path]] = None,
    save_dir: Union[str, Path] = "outputs/plots",
    show: bool = False
) -> Path:
    """
    Plot combined training curves (Losses + Accuracy & F1 Metrics) and save as a high-resolution image.

    Args:
        history: Dictionary containing history data, or path to history.json.
                 If None, automatically searches standard local and Google Drive paths.
        save_dir: Directory where the output plots will be saved.
        show: If True, calls plt.show().

    Returns:
        Path to the saved figure.
    """
    if history is None or isinstance(history, (str, Path)):
        history_data = load_history(history)
    else:
        history_data = history

    epochs = history_data.get("epochs", [])
    if not epochs:
        raise ValueError("History contains no epochs data to plot.")

    train_loss_total = history_data.get("train_loss_total", [])
    val_loss_total = history_data.get("val_loss_total", [])
    train_loss_sent = history_data.get("train_loss_sentiment", [])
    val_loss_sent = history_data.get("val_loss_sentiment", [])
    train_loss_asp = history_data.get("train_loss_aspect", [])
    val_loss_asp = history_data.get("val_loss_aspect", [])

    val_sent_acc = history_data.get("val_sentiment_acc", [])
    val_sent_f1 = history_data.get("val_sentiment_f1", [])
    val_asp_micro = history_data.get("val_aspect_micro_f1", [])
    val_asp_macro = history_data.get("val_aspect_macro_f1", [])

    best_epoch = history_data.get("best_epoch")
    best_val_loss = history_data.get("best_val_loss")
    early_stopped = history_data.get("early_stopping_triggered", False)
    early_stop_epoch = history_data.get("early_stopped_at_epoch")

    # If best_epoch was not explicitly saved, compute from val_loss_total
    if best_epoch is None and val_loss_total:
        min_loss = min(val_loss_total)
        min_idx = val_loss_total.index(min_loss)
        best_epoch = epochs[min_idx]
        best_val_loss = min_loss

    # Use a clean style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax_loss, ax_metrics) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # ----------------------------------------------------
    # Panel 1: Loss Curves
    # ----------------------------------------------------
    if train_loss_total:
        ax_loss.plot(epochs, train_loss_total, label="Train Total Loss", color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
    if val_loss_total:
        ax_loss.plot(epochs, val_loss_total, label="Val Total Loss", color="#ff7f0e", linewidth=2.2, marker="s", markersize=4)
    if val_loss_sent:
        ax_loss.plot(epochs, val_loss_sent, label="Val Sentiment Loss", color="#2ca02c", linestyle="--", alpha=0.7)
    if val_loss_asp:
        ax_loss.plot(epochs, val_loss_asp, label="Val Aspect Loss", color="#9467bd", linestyle="--", alpha=0.7)

    # Mark Best Model Checkpoint
    if best_epoch is not None and best_val_loss is not None:
        ax_loss.scatter(
            [best_epoch], [best_val_loss],
            color="#d62728", s=200, zorder=5, marker="*",
            label=f"Best Model (Epoch {best_epoch} - Loss: {best_val_loss:.4f})"
        )
        ax_loss.annotate(
            f"  ⭐ Best Ep {best_epoch}\n  ({best_val_loss:.4f})",
            (best_epoch, best_val_loss),
            textcoords="offset points",
            xytext=(10, 10),
            fontweight="bold",
            color="#d62728",
            fontsize=9
        )

    # Mark Early Stopping Trigger
    if early_stopped and early_stop_epoch is not None:
        ax_loss.axvline(
            x=early_stop_epoch,
            color="#d62728",
            linestyle=":",
            linewidth=2.0,
            label=f"🛑 Early Stopped (Epoch {early_stop_epoch})"
        )

    ax_loss.set_title("Training & Validation Loss", fontsize=14, fontweight="bold", pad=12)
    ax_loss.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax_loss.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax_loss.set_xticks(epochs)
    ax_loss.legend(loc="upper right", frameon=True, fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    # ----------------------------------------------------
    # Panel 2: Validation Metrics Curves (Accuracy & F1)
    # ----------------------------------------------------
    if val_sent_acc:
        ax_metrics.plot(epochs, val_sent_acc, label="Sentiment Accuracy", color="#2ca02c", linewidth=2.0, marker="o", markersize=4)
    if val_sent_f1:
        ax_metrics.plot(epochs, val_sent_f1, label="Sentiment F1", color="#17becf", linewidth=2.0, marker="^", markersize=4)
    if val_asp_micro:
        ax_metrics.plot(epochs, val_asp_micro, label="Aspect Micro F1", color="#9467bd", linewidth=2.0, marker="s", markersize=4)
    if val_asp_macro:
        ax_metrics.plot(epochs, val_asp_macro, label="Aspect Macro F1", color="#8c564b", linewidth=2.0, marker="d", markersize=4)

    # Best epoch reference line on metrics panel
    if best_epoch is not None:
        ax_metrics.axvline(
            x=best_epoch,
            color="#2ca02c",
            linestyle="--",
            alpha=0.6,
            label=f"Best Model Epoch ({best_epoch})"
        )

    if early_stopped and early_stop_epoch is not None:
        ax_metrics.axvline(
            x=early_stop_epoch,
            color="#d62728",
            linestyle=":",
            linewidth=2.0,
            label=f"🛑 Early Stopped (Epoch {early_stop_epoch})"
        )

    ax_metrics.set_title("Validation Accuracy & F1 Scores", fontsize=14, fontweight="bold", pad=12)
    ax_metrics.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax_metrics.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax_metrics.set_xticks(epochs)
    ax_metrics.set_ylim([0.0, 1.05])
    ax_metrics.legend(loc="lower right", frameon=True, fontsize=9)
    ax_metrics.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to local outputs
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    out_file = save_path / "training_curves.png"
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Training curves saved locally to: {out_file}")

    # Copy to Google Drive if available
    try:
        gdrive_plots_dir = Path(GDRIVE_PLOTS_DIR)
        gdrive_plots_dir.mkdir(parents=True, exist_ok=True)
        gdrive_file = gdrive_plots_dir / "training_curves.png"
        shutil.copy2(out_file, gdrive_file)
        print(f"Training curves copied to Google Drive: {gdrive_file}")
    except Exception:
        pass

    if show:
        plt.show()

    plt.close(fig)
    return out_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot training curves from history.json")
    parser.add_argument("--history", type=str, default=None, help="Path to history.json")
    parser.add_argument("--save-dir", type=str, default="outputs/plots", help="Directory to save plot image")
    parser.add_argument("--show", action="store_true", help="Display plot window")
    args = parser.parse_args()

    plot_training_curves(history=args.history, save_dir=args.save_dir, show=args.show)

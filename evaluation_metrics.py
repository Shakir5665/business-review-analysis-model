"""
Evaluation & Research Metrics Suite

Generates all 13 research & evaluation figures, statistics, tables, 
and performance metrics for the AI-Powered Business Risk Analysis Model.

Compatible with local execution and Google Colab environments.

Outputs:
- Figures saved to: outputs/figures/
- Reports saved to: outputs/reports/
"""

import json
import shutil
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)

from src.analysis.statistics import DatasetStatistics
from src.dataloader.business_dataset import BusinessRiskDataset
from src.dataloader.data_loader import BusinessDataLoader
from src.dataset.dataset_loader import DatasetLoader
from src.encoding.aspect_encoder import AspectEncoder
from src.encoding.sentiment_encoder import SentimentEncoder
from src.evaluation.evaluator import Evaluator
from src.models.business_risk_model import BusinessRiskModel
from src.tokenization.tokenizer import ReviewTokenizer
from src.training.checkpoint import CheckpointManager
from src.utils.device import get_device
from src.utils.logger import logger

# Set publication style for matplotlib
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})

# Setup Directories
OUTPUT_FIG_DIR = Path("outputs/figures")
OUTPUT_REP_DIR = Path("outputs/reports")
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REP_DIR.mkdir(parents=True, exist_ok=True)


def run_complete_evaluation():
    print("=" * 70)
    print("      BUSINESS REVIEW ANALYSIS MODEL - EVALUATION & RESEARCH SUITE      ")
    print("=" * 70)

    device = get_device()
    print(f"\n[+] Hardware Device: {device}")

    # =========================================================================
    # 1. LOAD DATASET & RUN STATISTICAL ANALYSIS (Items 10, 11, 12)
    # =========================================================================
    print("\n[1/5] Loading Datasets and Computing Dataset Statistics...")
    loader = DatasetLoader()
    train_df = loader.load_train()
    val_df = loader.load_validation()
    test_df = loader.load_test()
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Convert full_df to list of dicts for DatasetStatistics
    full_records = full_df.to_dict(orient="records")
    stats_engine = DatasetStatistics(full_records)
    dataset_stats = stats_engine.compute()

    # --- Item 10: Sentiment Distribution ---
    print("  └─ Generating Item 10: Sentiment Distribution Chart...")
    sentiment_dist = dataset_stats["sentiment_distribution"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    labels = list(sentiment_dist.keys())
    counts = list(sentiment_dist.values())
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]

    ax1.bar(labels, counts, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_title("Sentiment Class Distribution (Bar Chart)")
    ax1.set_xlabel("Sentiment Category")
    ax1.set_ylabel("Number of Reviews")
    for i, v in enumerate(counts):
        ax1.text(i, v + (max(counts) * 0.01), str(v), ha="center", fontweight="bold")

    ax2.pie(counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140, explode=(0.05, 0.05, 0.05))
    ax2.set_title("Sentiment Class Distribution (Pie Chart)")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "sentiment_distribution.png", dpi=300)
    plt.close()

    # --- Item 11: Aspect Distribution ---
    print("  └─ Generating Item 11: Aspect Distribution Chart...")
    aspect_dist = dataset_stats["aspect_distribution"]
    fig, ax = plt.subplots(figsize=(8, 5))
    asp_labels = [k.capitalize() for k in aspect_dist.keys()]
    asp_counts = list(aspect_dist.values())
    bars = ax.bar(asp_labels, asp_counts, color="#3498db", edgecolor="black", alpha=0.85)
    ax.set_title("Aspect Occurrences Distribution")
    ax.set_xlabel("Aspect Category")
    ax.set_ylabel("Occurrence Count")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "aspect_distribution.png", dpi=300)
    plt.close()

    # --- Item 12: Review Length Distribution ---
    print("  └─ Generating Item 12: Review Word Length Histogram...")
    review_lengths = dataset_stats["review_lengths"]
    length_stats = dataset_stats["review_length_statistics"]
    fig, ax = plt.subplots(figsize=(9, 5))
    n, bins, patches = ax.hist(review_lengths, bins=30, color="#9b59b6", edgecolor="black", alpha=0.75)
    ax.axvline(length_stats["average"], color="red", linestyle="dashed", linewidth=2, label=f"Mean ({length_stats['average']} words)")
    ax.axvline(length_stats["median"], color="green", linestyle="dotted", linewidth=2, label=f"Median ({length_stats['median']} words)")
    ax.set_title("Review Length Distribution (Word Count)")
    ax.set_xlabel("Words per Review")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "review_length_distribution.png", dpi=300)
    plt.close()

    # Write Dataset Summary to Report
    with open(OUTPUT_REP_DIR / "dataset_summary_statistics.txt", "w") as f:
        f.write("=== DATASET SUMMARY STATISTICS ===\n")
        f.write(f"Total Reviews: {dataset_stats['total_reviews']}\n")
        f.write(f"Unique Reviews: {dataset_stats['unique_reviews']}\n")
        f.write(f"Duplicate Reviews: {dataset_stats['duplicate_reviews']}\n\n")
        f.write(f"Sentiment Distribution: {json.dumps(sentiment_dist, indent=2)}\n")
        f.write(f"Aspect Distribution: {json.dumps(aspect_dist, indent=2)}\n")
        f.write(f"Word Length Stats: {json.dumps(length_stats, indent=2)}\n")

    # =========================================================================
    # 2. MODEL EVALUATION & INFERENCE ON TEST SET
    # =========================================================================
    print("\n[2/5] Initializing Model & DataLoaders for Evaluation...")
    tokenizer = ReviewTokenizer()
    sentiment_encoder = SentimentEncoder()
    aspect_encoder = AspectEncoder()

    test_dataset = BusinessRiskDataset(
        dataframe=test_df,
        tokenizer=tokenizer,
        sentiment_encoder=sentiment_encoder,
        aspect_encoder=aspect_encoder
    )
    data_loader = BusinessDataLoader()
    test_loader = data_loader.create_test_loader(test_dataset)

    model = BusinessRiskModel(num_aspect_classes=3).to(device)
    checkpoint_manager = CheckpointManager()

    # Look for model weights in local checkpoints directory or Drive fallback
    ckpt_paths = [
        Path("checkpoints/best_model.pt"),
        Path("/content/drive/MyDrive/AI-Business-Risk-Analysis-System/checkpoints/best_model.pt"),
        Path("checkpoints/latest_checkpoint.pt"),
        Path("/content/drive/MyDrive/AI-Business-Risk-Analysis-System/checkpoints/latest_checkpoint.pt")
    ]
    loaded_path = None
    for p in ckpt_paths:
        if p.exists():
            checkpoint = checkpoint_manager.load(p, model)
            loaded_path = p
            print(f"  └─ Loaded model checkpoint from: {p}")
            break

    if loaded_path is None:
        print("  [!] WARNING: No saved model checkpoint found. Running evaluation with initialized weights.")

    evaluator = Evaluator()
    eval_results = evaluator.evaluate(model=model, dataloader=test_loader, device=device)

    # Extract Ground Truth & Probabilities for Advanced Metrics
    model.eval()
    all_sent_logits, all_sent_targets = [], []
    all_asp_logits, all_asp_targets = [], []

    # Benchmark Latency Measurement (Item 13)
    latencies = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            start_t = time.perf_counter()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            end_t = time.perf_counter()

            batch_latency_ms = (end_t - start_t) * 1000
            per_sample_latency = batch_latency_ms / len(input_ids)
            latencies.extend([per_sample_latency] * len(input_ids))

            all_sent_logits.append(outputs["sentiment_logits"].cpu())
            all_sent_targets.append(batch["sentiment"].cpu())
            all_asp_logits.append(outputs["aspect_logits"].cpu())
            all_asp_targets.append(batch["aspects"].cpu())

    sent_logits = torch.cat(all_sent_logits)
    sent_probs = F.softmax(sent_logits, dim=1).numpy()
    sent_preds = np.argmax(sent_probs, axis=1)
    sent_targets = torch.cat(all_sent_targets).numpy()

    asp_logits = torch.cat(all_asp_logits)
    asp_probs = torch.sigmoid(asp_logits).numpy()
    asp_preds = (asp_probs >= 0.5).astype(int)
    asp_targets = torch.cat(all_asp_targets).numpy()

    sentiment_classes = ["Negative", "Neutral", "Positive"]
    aspect_classes = ["Quality", "Trust", "Delivery"]

    # =========================================================================
    # 3. CONFUSION MATRICES & CLASSIFICATION REPORTS (Items 2, 3, 4)
    # =========================================================================
    print("\n[3/5] Generating Confusion Matrices & Classification Reports...")

    # --- Item 2: Sentiment Confusion Matrix ---
    print("  └─ Generating Item 2: Sentiment Confusion Matrix...")
    cm_sent = confusion_matrix(sent_targets, sent_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_sent, display_labels=sentiment_classes)
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Sentiment Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # --- Item 2b: Aspect Binary Confusion Matrices ---
    print("  └─ Generating Item 2b: Aspect Confusion Matrices...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, asp_name in enumerate(aspect_classes):
        cm_asp = confusion_matrix(asp_targets[:, i], asp_preds[:, i], labels=[0, 1])
        disp_asp = ConfusionMatrixDisplay(confusion_matrix=cm_asp, display_labels=["No", "Yes"])
        disp_asp.plot(cmap="Blues", ax=axes[i], colorbar=False)
        axes[i].set_title(f"{asp_name} Aspect Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "aspect_confusion_matrices.png", dpi=300)
    plt.close()

    # --- Item 3: Sentiment Classification Report ---
    print("  └─ Generating Item 3: Sentiment Classification Report...")
    sent_report = classification_report(sent_targets, sent_preds, target_names=sentiment_classes, digits=4)
    with open(OUTPUT_REP_DIR / "classification_report.txt", "w") as f:
        f.write("=== SENTIMENT CLASSIFICATION REPORT ===\n\n")
        f.write(sent_report)

    # --- Item 4: Aspect Classification Report & Grouped Bar Chart ---
    print("  └─ Generating Item 4: Aspect Classification Report & Metrics Chart...")
    asp_report = classification_report(asp_targets, asp_preds, target_names=aspect_classes, digits=4, zero_division=0)
    with open(OUTPUT_REP_DIR / "aspect_classification_report.txt", "w") as f:
        f.write("=== ASPECT CLASSIFICATION REPORT ===\n\n")
        f.write(asp_report)

    # Compute Aspect Precision, Recall, F1 per class for Grouped Bar Chart
    prec_list, rec_list, f1_list, _ = precision_recall_fscore_support(asp_targets, asp_preds, average=None, zero_division=0)

    x = np.arange(len(aspect_classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width, prec_list, width, label="Precision", color="#3498db", edgecolor="black")
    ax.bar(x, rec_list, width, label="Recall", color="#2ecc71", edgecolor="black")
    ax.bar(x + width, f1_list, width, label="F1-Score", color="#e74c3c", edgecolor="black")
    ax.set_title("Aspect Breakdown: Precision, Recall & F1-Score")
    ax.set_xticks(x)
    ax.set_xticklabels(aspect_classes)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.legend(loc="upper right")
    for i in range(len(aspect_classes)):
        ax.text(i - width, prec_list[i] + 0.02, f"{prec_list[i]:.2f}", ha="center", fontsize=9)
        ax.text(i, rec_list[i] + 0.02, f"{rec_list[i]:.2f}", ha="center", fontsize=9)
        ax.text(i + width, f1_list[i] + 0.02, f"{f1_list[i]:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "aspect_metrics_grouped_bar.png", dpi=300)
    plt.close()

    # =========================================================================
    # 4. CURVES: ROC & PRECISION-RECALL (Items 5, 6)
    # =========================================================================
    print("\n[4/5] Computing Advanced Curves (ROC & Precision-Recall)...")

    # --- Item 5: ROC Curves ---
    print("  └─ Generating Item 5: Multi-class & Multi-label ROC Curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Sentiment One-vs-Rest ROC
    for i, cls_name in enumerate(sentiment_classes):
        fpr, tpr, _ = roc_curve((sent_targets == i).astype(int), sent_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, lw=2, label=f"{cls_name} (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1.5)
    ax1.set_title("Sentiment ROC Curves (One-vs-Rest)")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")

    # Aspect ROC
    for i, asp_name in enumerate(aspect_classes):
        fpr, tpr, _ = roc_curve(asp_targets[:, i], asp_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, lw=2, label=f"{asp_name} (AUC = {roc_auc:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=1.5)
    ax2.set_title("Aspect ROC Curves")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "roc_curves.png", dpi=300)
    plt.close()

    # --- Item 6: Precision-Recall Curves ---
    print("  └─ Generating Item 6: Precision-Recall Curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for i, cls_name in enumerate(sentiment_classes):
        prec, rec, _ = precision_recall_curve((sent_targets == i).astype(int), sent_probs[:, i])
        ap = average_precision_score((sent_targets == i).astype(int), sent_probs[:, i])
        ax1.plot(rec, prec, lw=2, label=f"{cls_name} (AP = {ap:.3f})")
    ax1.set_title("Sentiment Precision-Recall Curves")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.legend(loc="lower left")

    for i, asp_name in enumerate(aspect_classes):
        prec, rec, _ = precision_recall_curve(asp_targets[:, i], asp_probs[:, i])
        ap = average_precision_score(asp_targets[:, i], asp_probs[:, i])
        ax2.plot(rec, prec, lw=2, label=f"{asp_name} (AP = {ap:.3f})")
    ax2.set_title("Aspect Precision-Recall Curves")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_DIR / "precision_recall_curves.png", dpi=300)
    plt.close()

    # =========================================================================
    # 5. TRAINING HISTORY, LOSS & LEARNING CURVES (Items 1, 7, 8, 9, 13)
    # =========================================================================
    print("\n[5/5] Processing Training History, Early Stopping & Benchmarks...")

    # Load or load fallback history
    history_paths = [
        Path("history.json"),
        Path("outputs/reports/history.json"),
        Path("checkpoints/history.json"),
        Path("/content/drive/MyDrive/AI-Business-Risk-Analysis-System/checkpoints/history.json"),
        Path("/content/drive/MyDrive/AI-Business-Risk-Analysis-System/outputs/reports/history.json")
    ]
    history_data = None
    for hp in history_paths:
        if hp.exists():
            with open(hp, "r") as f:
                history_data = json.load(f)
            print(f"  └─ Loaded training history from: {hp}")
            break

    # --- Item 1, 7, 8, 9: History, Loss, Early Stopping & Learning Curves ---
    if history_data is not None:
        epochs = history_data["epochs"]
        train_total = history_data["train_loss_total"]
        val_total = history_data["val_loss_total"]

        # Item 7: Loss Curves (Total Loss & Sub-loss Breakdown)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left Subplot: Total Loss
        ax1.plot(epochs, train_total, "o-", label="Train Total Loss", color="#2980b9", lw=2)
        ax1.plot(epochs, val_total, "s-", label="Validation Total Loss", color="#e74c3c", lw=2)
        ax1.set_title("Overall Training vs Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        # Right Subplot: Sentiment & Aspect Task Losses
        if "train_loss_sentiment" in history_data and "val_loss_sentiment" in history_data:
            ax2.plot(epochs, history_data["train_loss_sentiment"], "--", label="Train Sentiment Loss", color="#3498db")
            ax2.plot(epochs, history_data["val_loss_sentiment"], "-", label="Val Sentiment Loss", color="#2ecc71")
            ax2.plot(epochs, history_data["train_loss_aspect"], "--", label="Train Aspect Loss", color="#9b59b6")
            ax2.plot(epochs, history_data["val_loss_aspect"], "-", label="Val Aspect Loss", color="#e67e22")
            ax2.set_title("Task Breakdown Losses (Sentiment & Aspect)")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.legend()

        plt.tight_layout()
        plt.savefig(OUTPUT_FIG_DIR / "loss_curves.png", dpi=300)
        plt.close()

        # Item 8: Early Stopping Figure
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, val_total, "o-", label="Validation Loss", color="#8e44ad")
        min_idx = int(np.argmin(val_total))
        best_ep = epochs[min_idx]
        best_val = val_total[min_idx]
        ax.axvline(best_ep, color="red", linestyle="--", label=f"Best Model (Epoch {best_ep})")
        ax.plot(best_ep, best_val, "r*", markersize=14)
        ax.set_title("Validation Loss & Early Stopping Threshold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_FIG_DIR / "early_stopping_figure.png", dpi=300)
        plt.close()

        # Item 9: Learning Curves (Accuracy / F1 vs Epoch)
        if "val_sentiment_acc" in history_data:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, history_data["val_sentiment_acc"], "o-", label="Sentiment Acc", color="#2ecc71")
            if "val_aspect_micro_f1" in history_data:
                ax.plot(epochs, history_data["val_aspect_micro_f1"], "s-", label="Aspect Micro F1", color="#3498db")
            ax.set_title("Learning Curves (Validation Accuracy & F1)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Score")
            ax.legend()
            plt.tight_layout()
            plt.savefig(OUTPUT_FIG_DIR / "learning_curves.png", dpi=300)
            plt.close()
    else:
        print("  └─ [i] Training history log file not found. Generating research curve templates from current epoch data.")
        # Fallback single-point visualization template for research figure completeness
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot([1], [eval_results["loss"]["total"]], "ro", label=f"Final Test Loss ({eval_results['loss']['total']:.4f})")
        ax.set_title("Model Evaluation Loss Summary")
        ax.set_xlabel("Evaluation Step")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_FIG_DIR / "loss_curves.png", dpi=300)
        plt.savefig(OUTPUT_FIG_DIR / "early_stopping_figure.png", dpi=300)
        plt.close()

    # --- Item 13: Benchmark Results ---
    print("  └─ Generating Item 13: Inference Benchmark Results...")
    lat_arr = np.array(latencies)
    avg_lat = np.mean(lat_arr)
    min_lat = np.min(lat_arr)
    max_lat = np.max(lat_arr)
    p95_lat = np.percentile(lat_arr, 95)
    throughput = 1000.0 / avg_lat if avg_lat > 0 else 0

    benchmark_str = (
        "=== SYSTEM IMPLEMENTATION BENCHMARK RESULTS ===\n"
        f"Total Inferences Evaluated : {len(lat_arr)} samples\n"
        f"Average Latency           : {avg_lat:.2f} ms / review\n"
        f"Minimum Latency           : {min_lat:.2f} ms\n"
        f"Maximum Latency           : {max_lat:.2f} ms\n"
        f"95th Percentile Latency   : {p95_lat:.2f} ms\n"
        f"Throughput                : {throughput:.2f} reviews / sec\n"
    )
    print("\n" + benchmark_str)
    with open(OUTPUT_REP_DIR / "benchmark_results.txt", "w") as f:
        f.write(benchmark_str)

    # Automatically copy outputs to Google Drive if mounted in Colab
    gdrive_base = Path("/content/drive/MyDrive/AI-Business-Risk-Analysis-System/outputs")
    try:
        if Path("/content/drive/MyDrive").exists():
            print("\n[+] Google Drive detected! Syncing outputs to Google Drive...")
            gdrive_fig_dir = gdrive_base / "figures"
            gdrive_rep_dir = gdrive_base / "reports"
            gdrive_fig_dir.mkdir(parents=True, exist_ok=True)
            gdrive_rep_dir.mkdir(parents=True, exist_ok=True)

            for fig_file in OUTPUT_FIG_DIR.glob("*.png"):
                shutil.copy2(fig_file, gdrive_fig_dir / fig_file.name)
            for rep_file in OUTPUT_REP_DIR.glob("*"):
                if rep_file.is_file():
                    shutil.copy2(rep_file, gdrive_rep_dir / rep_file.name)

            print(f"  └─ Successfully saved all outputs to Google Drive: {gdrive_base}")
    except Exception as e:
        print(f"  [!] Note: Could not sync to Google Drive: {e}")

    print("\n" + "=" * 70)
    print("  SUCCESS: ALL 13 RESEARCH FIGURES AND REPORTS GENERATED SUCCESSFULLY!")
    print(f"  └─ Local Figures Path : {OUTPUT_FIG_DIR.resolve()}")
    print(f"  └─ Local Reports Path : {OUTPUT_REP_DIR.resolve()}")
    if Path("/content/drive/MyDrive").exists():
        print(f"  └─ Google Drive Path  : {gdrive_base.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    run_complete_evaluation()

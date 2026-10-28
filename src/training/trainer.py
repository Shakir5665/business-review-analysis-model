"""
Trainer

Handles the complete training pipeline.

Project:
AI-Powered Business Risk Analysis
and Recommendation System
"""

import json
from pathlib import Path

import torch
import torch.nn as nn

from configs.training_config import (
    NUM_EPOCHS,
    GRADIENT_CLIP,
    EARLY_STOPPING,
    MIN_DELTA,
    GDRIVE_CHECKPOINT_DIR,
    GDRIVE_REPORTS_DIR
)

from src.training.early_stopping import EarlyStopping

from src.training.losses import LossComputer
from src.training.validator import Validator
from src.training.checkpoint import CheckpointManager

from src.utils.logger import logger


class Trainer:
    """
    Trainer for BusinessRiskModel.
    """

    def __init__(
        self,
        model,
        train_loader,
        validation_loader,
        optimizer,
        scheduler,
        loss_computer,
        device
    ):

        self.model = model

        self.train_loader = train_loader

        self.validation_loader = validation_loader

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.device = device

        self.loss_computer = loss_computer

        self.validator = Validator()

        self.checkpoint_manager = CheckpointManager()

        self.start_epoch = 1

        # -----------------------------
        # Early Stopping
        # -----------------------------

        self.early_stopping = None

        if EARLY_STOPPING:

            self.early_stopping = EarlyStopping()

        self.num_epochs = NUM_EPOCHS

        self.gradient_clip = GRADIENT_CLIP

        self.min_delta = MIN_DELTA

        self.best_validation_loss = float("inf")
        self.best_epoch = None

        self.history = {
            "epochs": [],
            "train_loss_total": [],
            "train_loss_sentiment": [],
            "train_loss_aspect": [],
            "val_loss_total": [],
            "val_loss_sentiment": [],
            "val_loss_aspect": [],
            "val_sentiment_acc": [],
            "val_sentiment_f1": [],
            "val_aspect_micro_f1": [],
            "val_aspect_macro_f1": [],
            "is_best_epoch": [],
            "best_val_loss_so_far": [],
            "early_stopping_counter": [],
            "early_stopping_triggered": False,
            "early_stopped_at_epoch": None,
            "best_epoch": None,
            "best_val_loss": None
        }

        logger.info("Trainer initialized.")

    def _save_history(self):
        """Save history.json to local directories and Google Drive."""
        target_paths = [
            Path("outputs/reports/history.json"),
            Path("checkpoints/history.json"),
            Path(f"{GDRIVE_REPORTS_DIR}/history.json"),
            Path(f"{GDRIVE_CHECKPOINT_DIR}/history.json")
        ]
        for p in target_paths:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "w") as f:
                    json.dump(self.history, f, indent=2)
            except Exception:
                pass

    def load_history(self):
        """Load and synchronize existing training history when resuming."""
        history_paths = [
            Path("outputs/reports/history.json"),
            Path("checkpoints/history.json"),
            Path(f"{GDRIVE_REPORTS_DIR}/history.json"),
            Path(f"{GDRIVE_CHECKPOINT_DIR}/history.json")
        ]
        list_keys = [
            "epochs",
            "train_loss_total",
            "train_loss_sentiment",
            "train_loss_aspect",
            "val_loss_total",
            "val_loss_sentiment",
            "val_loss_aspect",
            "val_sentiment_acc",
            "val_sentiment_f1",
            "val_aspect_micro_f1",
            "val_aspect_macro_f1",
            "is_best_epoch",
            "best_val_loss_so_far",
            "early_stopping_counter"
        ]
        for hp in history_paths:
            if hp.exists():
                try:
                    with open(hp, "r") as f:
                        loaded = json.load(f)
                        if isinstance(loaded, dict) and "epochs" in loaded:
                            # Trim entries >= start_epoch to avoid duplicates when resuming
                            valid_indices = [i for i, ep in enumerate(loaded["epochs"]) if ep < self.start_epoch]
                            for k in list_keys:
                                if k in loaded and isinstance(loaded[k], list):
                                    self.history[k] = [loaded[k][i] for i in valid_indices if i < len(loaded[k])]
                            
                            # Restore best validation loss and best epoch from loaded history
                            if len(self.history["val_loss_total"]) > 0:
                                min_val_loss = min(self.history["val_loss_total"])
                                min_idx = self.history["val_loss_total"].index(min_val_loss)
                                self.best_validation_loss = min_val_loss
                                self.best_epoch = self.history["epochs"][min_idx]
                                self.history["best_val_loss"] = min_val_loss
                                self.history["best_epoch"] = self.best_epoch
                                if self.early_stopping is not None:
                                    self.early_stopping.best_loss = min_val_loss
                                    if len(self.history["early_stopping_counter"]) > 0:
                                        self.early_stopping.counter = self.history["early_stopping_counter"][-1]

                            self.history["early_stopping_triggered"] = False
                            self.history["early_stopped_at_epoch"] = None

                            logger.info(f"Loaded existing training history up to Epoch {self.start_epoch - 1} from {hp}")
                            break
                except Exception as e:
                    logger.warning(f"Could not load history from {hp}: {e}")
    
        # --------------------------------------------------

    def train_epoch(self):

        self.model.train()

        total_sentiment_loss = 0.0

        total_aspect_loss = 0.0

        total_loss = 0.0

        for batch in self.train_loader:

            input_ids = batch["input_ids"].to(self.device)

            attention_mask = batch["attention_mask"].to(self.device)

            sentiment = batch["sentiment"].to(self.device)

            aspects = batch["aspects"].to(self.device)

            # -----------------------------
            # Forward Pass
            # -----------------------------

            outputs = self.model(

                input_ids=input_ids,

                attention_mask=attention_mask

            )

            # -----------------------------
            # Compute Loss
            # -----------------------------

            losses = self.loss_computer.compute(

                outputs,

                sentiment,

                aspects

            )

            # -----------------------------
            # Backpropagation
            # -----------------------------

            self.optimizer.zero_grad()

            losses["total_loss"].backward()

            # -----------------------------
            # Gradient Clipping
            # -----------------------------

            torch.nn.utils.clip_grad_norm_(

                self.model.parameters(),

                self.gradient_clip

            )

            # -----------------------------
            # Update Parameters
            # -----------------------------

            self.optimizer.step()

            self.scheduler.step()

            # -----------------------------
            # Statistics
            # -----------------------------

            total_sentiment_loss += losses["sentiment_loss"].item()

            total_aspect_loss += losses["aspect_loss"].item()

            total_loss += losses["total_loss"].item()

        num_batches = len(self.train_loader)

        return {

            "sentiment": total_sentiment_loss / num_batches,

            "aspect": total_aspect_loss / num_batches,

            "total": total_loss / num_batches

        }
    
        # --------------------------------------------------

    def train(self):

        logger.info("Training started.")

        for epoch in range(self.start_epoch, self.num_epochs + 1):

            print()

            print("=" * 60)

            print(f"Epoch {epoch}/{self.num_epochs}")

            print("=" * 60)

            # -----------------------------
            # Training
            # -----------------------------

            train_loss = self.train_epoch()

            # -----------------------------
            # Validation
            # -----------------------------

            validation_results = self.validator.validate(

                model=self.model,

                dataloader=self.validation_loader,

                device=self.device

            )

            validation_loss = validation_results["loss"]["total"]

            # -----------------------------
            # Save Latest Checkpoint
            # -----------------------------

            self.checkpoint_manager.save_latest(

                model=self.model,

                optimizer=self.optimizer,

                scheduler=self.scheduler,

                epoch=epoch

            )

            # -----------------------------
            # Save Best Model
            # -----------------------------

            is_best = False
            if validation_loss < (self.best_validation_loss - self.min_delta):

                self.best_validation_loss = validation_loss
                self.best_epoch = epoch

                self.checkpoint_manager.save_best(

                    model=self.model,

                    optimizer=self.optimizer,

                    scheduler=self.scheduler,

                    epoch=epoch,

                    validation_loss=validation_loss

                )

                is_best = True
                best_saved = "YES"

            else:

                best_saved = "NO"

            # -----------------------------
            # Early Stopping Check
            # -----------------------------

            early_stop_triggered = False
            es_counter = 0

            if self.early_stopping is not None:
                if is_best:
                    self.early_stopping.best_loss = self.best_validation_loss
                    self.early_stopping.counter = 0
                    early_stop_triggered = False
                else:
                    early_stop_triggered = self.early_stopping.should_stop(validation_loss)
                es_counter = getattr(self.early_stopping, "counter", 0)

            # -----------------------------
            # Epoch Summary
            # -----------------------------

            print()

            print("Training Loss")

            print(f"  Sentiment : {train_loss['sentiment']:.4f}")

            print(f"  Aspect    : {train_loss['aspect']:.4f}")

            print(f"  Total     : {train_loss['total']:.4f}")

            print()

            print("Validation Loss")

            print(f"  Sentiment : {validation_results['loss']['sentiment']:.4f}")

            print(f"  Aspect    : {validation_results['loss']['aspect']:.4f}")

            print(f"  Total     : {validation_results['loss']['total']:.4f}")

            print()

            print("Sentiment Metrics")

            print(f"  Accuracy  : {validation_results['sentiment']['accuracy']:.4f}")

            print(f"  Precision : {validation_results['sentiment']['precision']:.4f}")

            print(f"  Recall    : {validation_results['sentiment']['recall']:.4f}")

            print(f"  F1 Score  : {validation_results['sentiment']['f1']:.4f}")

            print()

            print("Aspect Metrics")

            print(f"  Micro F1  : {validation_results['aspect']['micro_f1']:.4f}")

            print(f"  Macro F1  : {validation_results['aspect']['macro_f1']:.4f}")

            print()

            best_epoch_display = self.best_epoch if self.best_epoch is not None else epoch
            print(f"Best Model Saved       : {best_saved} (Best val_loss: {self.best_validation_loss:.4f} @ Epoch {best_epoch_display})")
            if self.early_stopping is not None:
                print(f"Early Stopping Counter : {es_counter}/{self.early_stopping.patience}")

            # -----------------------------
            # Save Training History
            # -----------------------------
            self.history["epochs"].append(epoch)
            self.history["train_loss_total"].append(train_loss['total'])
            self.history["train_loss_sentiment"].append(train_loss['sentiment'])
            self.history["train_loss_aspect"].append(train_loss['aspect'])
            self.history["val_loss_total"].append(validation_results['loss']['total'])
            self.history["val_loss_sentiment"].append(validation_results['loss']['sentiment'])
            self.history["val_loss_aspect"].append(validation_results['loss']['aspect'])
            self.history["val_sentiment_acc"].append(validation_results['sentiment']['accuracy'])
            self.history["val_sentiment_f1"].append(validation_results['sentiment']['f1'])
            self.history["val_aspect_micro_f1"].append(validation_results['aspect']['micro_f1'])
            self.history["val_aspect_macro_f1"].append(validation_results['aspect']['macro_f1'])
            self.history["is_best_epoch"].append(is_best)
            self.history["best_val_loss_so_far"].append(self.best_validation_loss)
            self.history["early_stopping_counter"].append(es_counter)
            self.history["best_epoch"] = self.best_epoch
            self.history["best_val_loss"] = self.best_validation_loss

            if early_stop_triggered:
                self.history["early_stopping_triggered"] = True
                self.history["early_stopped_at_epoch"] = epoch
                self._save_history()

                print()
                print("=" * 60)
                print("🛑 Early Stopping Triggered")
                print(f"Training stopped at Epoch {epoch} (Best was Epoch {self.best_epoch} with val_loss: {self.best_validation_loss:.4f})")
                print("=" * 60)
                break

            self._save_history()

        logger.info("Training completed.")
    
    
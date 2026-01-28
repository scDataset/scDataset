"""
Multi-task trainer for Tahoe-100M experiments.

This module provides training and evaluation functionality for multi-task
classification models that predict cell_line, drug, moa_broad, and moa_fine
simultaneously.
"""

import os
import pickle
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from training_experiments.models.linear import TASK_NAMES, MultiTaskLoss


class MultiTaskTrainer:
    """
    Trainer for multi-task classification models.

    Parameters
    ----------
    model : nn.Module
        Multi-task model to train
    device : str
        Device to use for training ('cuda' or 'cpu')
    task_weights : dict, optional
        Weights for each task loss
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        task_weights: Dict[str, float] = None,
    ):
        self.model = model
        self.device = device
        self.task_names = TASK_NAMES

        # Move model to device
        self.model.to(self.device)

        # Loss function
        self.criterion = MultiTaskLoss(task_weights=task_weights)

        # Training history
        self.train_history: List[Dict[str, float]] = []
        self.val_history: List[Dict[str, float]] = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        log_interval: int = 100,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        optimizer : torch.optim.Optimizer
            Optimizer
        epoch : int
            Current epoch number
        log_interval : int
            Print progress every N batches

        Returns
        -------
        dict
            Training metrics for this epoch
        """
        self.model.train()

        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.task_names}
        task_correct = {task: 0 for task in self.task_names}
        task_total = {task: 0 for task in self.task_names}

        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Extract features and move to device
            features = batch["X"].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(features)

            # Compute loss
            loss, individual_losses = self.criterion(outputs, batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            for task_name in self.task_names:
                task_losses[task_name] += individual_losses[task_name].item()

            # Compute accuracy for each task
            with torch.no_grad():
                for task_name in self.task_names:
                    task_preds = torch.argmax(outputs[task_name], dim=1)
                    task_targets = batch[task_name].to(self.device)
                    task_correct[task_name] += (task_preds == task_targets).sum().item()
                    task_total[task_name] += task_targets.size(0)

            num_batches += 1

            # Progress logging
            if log_interval > 0 and batch_idx % log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}, Time: {elapsed:.1f}s"
                )

        # Compute average metrics
        metrics = {
            "total_loss": total_loss / num_batches,
            "time": time.time() - start_time,
        }

        # Add task-specific metrics
        for task_name in self.task_names:
            metrics[f"{task_name}_loss"] = task_losses[task_name] / num_batches
            if task_total[task_name] > 0:
                metrics[f"{task_name}_accuracy"] = (
                    task_correct[task_name] / task_total[task_name]
                )
            else:
                metrics[f"{task_name}_accuracy"] = 0.0

        self.train_history.append(metrics)
        return metrics

    def evaluate(self, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate the model.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader

        Returns
        -------
        dict
            Evaluation metrics including accuracy and F1 for each task
        """
        self.model.eval()

        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.task_names}

        # Collect all predictions and targets
        all_predictions = {task: [] for task in self.task_names}
        all_targets = {task: [] for task in self.task_names}

        num_batches = 0
        start_time = time.time()

        with torch.no_grad():
            for batch in val_loader:
                # Extract features and move to device
                features = batch["X"].to(self.device)

                # Forward pass
                outputs = self.model(features)

                # Compute loss
                loss, individual_losses = self.criterion(outputs, batch)

                # Update metrics
                total_loss += loss.item()
                for task_name in self.task_names:
                    task_losses[task_name] += individual_losses[task_name].item()

                # Collect predictions and targets
                for task_name in self.task_names:
                    task_preds = torch.argmax(outputs[task_name], dim=1)
                    task_targets = batch[task_name]

                    all_predictions[task_name].extend(task_preds.cpu().numpy())
                    all_targets[task_name].extend(task_targets.cpu().numpy())

                num_batches += 1

        # Compute metrics
        metrics = {
            "total_loss": total_loss / max(num_batches, 1),
            "time": time.time() - start_time,
        }

        # Add task-specific metrics
        for task_name in self.task_names:
            preds = np.array(all_predictions[task_name])
            targets = np.array(all_targets[task_name])

            if len(preds) > 0:
                metrics[f"{task_name}_loss"] = task_losses[task_name] / num_batches
                metrics[f"{task_name}_accuracy"] = accuracy_score(targets, preds)
                metrics[f"{task_name}_f1_macro"] = f1_score(
                    targets, preds, average="macro", zero_division=0
                )
                metrics[f"{task_name}_f1_weighted"] = f1_score(
                    targets, preds, average="weighted", zero_division=0
                )
            else:
                metrics[f"{task_name}_loss"] = 0.0
                metrics[f"{task_name}_accuracy"] = 0.0
                metrics[f"{task_name}_f1_macro"] = 0.0
                metrics[f"{task_name}_f1_weighted"] = 0.0

        self.val_history.append(metrics)
        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 1,
        learning_rate: float = 1e-5,
        log_interval: int = 1000,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the model for multiple epochs.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        num_epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate
        log_interval : int
            Print progress every N batches (0 to disable)
        save_dir : str, optional
            Directory to save model checkpoints

        Returns
        -------
        dict
            Training results including history
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        best_val_loss = float("inf")
        best_epoch = 0

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_metrics = self.train_epoch(
                train_loader, optimizer, epoch, log_interval
            )

            print(f"\nTraining: Loss={train_metrics['total_loss']:.4f}")
            for task_name in self.task_names:
                acc = train_metrics.get(f"{task_name}_accuracy", 0)
                print(f"  {task_name}: Acc={acc:.4f}")

            # Evaluate
            val_metrics = self.evaluate(val_loader)

            print(f"\nValidation: Loss={val_metrics['total_loss']:.4f}")
            for task_name in self.task_names:
                acc = val_metrics.get(f"{task_name}_accuracy", 0)
                f1 = val_metrics.get(f"{task_name}_f1_macro", 0)
                print(f"  {task_name}: Acc={acc:.4f}, F1={f1:.4f}")

            # Save best model
            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                best_epoch = epoch

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    model_path = os.path.join(save_dir, "best_model.pt")
                    torch.save(self.model.state_dict(), model_path)

        # Save final model
        if save_dir:
            final_path = os.path.join(save_dir, "final_model.pt")
            torch.save(self.model.state_dict(), final_path)

        return {
            "train_history": self.train_history,
            "val_history": self.val_history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save results to a pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(results, f)


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)

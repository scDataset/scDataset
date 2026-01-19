"""
Multi-task linear classifier for Tahoe-100M experiments.

This module provides a simple multi-task linear classifier that predicts
all 4 tasks simultaneously: cell_line, drug, moa_broad, moa_fine.

We use a linear model to avoid confounding effects from model
selection or hyperparameter tuning. Since training is I/O bottlenecked
rather than compute-bound, we combine all 4 linear layers for efficiency.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Task names in fixed order
TASK_NAMES = ["cell_line", "drug", "moa_broad", "moa_fine"]


class MultiTaskLinearClassifier(nn.Module):
    """
    Multi-task linear classifier that predicts all 4 tasks simultaneously.

    The model consists of 4 separate linear layers (one per task) applied
    directly to the gene expression features. No hidden layers or non-linearities
    are used to keep the model simple and avoid confounding effects.

    Parameters
    ----------
    input_dim : int
        Input feature dimension (number of genes)
    task_dims : dict
        Dictionary with number of classes for each task:
        {'cell_line': n_cell_lines, 'drug': n_drugs,
         'moa_broad': n_moa_broad, 'moa_fine': n_moa_fine}

    Attributes
    ----------
    task_heads : nn.ModuleDict
        Dictionary of linear layers for each task
    """

    def __init__(self, input_dim: int, task_dims: Dict[str, int]):
        super().__init__()

        self.input_dim = input_dim
        self.task_dims = task_dims
        self.task_names = TASK_NAMES

        # Task-specific linear layers
        self.task_heads = nn.ModuleDict()
        for task_name in self.task_names:
            if task_name not in task_dims:
                raise ValueError(f"task_dims must contain '{task_name}'")
            self.task_heads[task_name] = nn.Linear(input_dim, task_dims[task_name])

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch_size, input_dim)

        Returns
        -------
        dict
            Dictionary with logits for each task:
            {'cell_line': logits, 'drug': logits,
             'moa_broad': logits, 'moa_fine': logits}
        """
        outputs = {}
        for task_name in self.task_names:
            outputs[task_name] = self.task_heads[task_name](x)
        return outputs

    def get_predictions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get class predictions (argmax of logits).

        Parameters
        ----------
        x : torch.Tensor
            Input features

        Returns
        -------
        dict
            Dictionary with predicted class indices for each task
        """
        logits = self.forward(x)
        return {
            task_name: torch.argmax(task_logits, dim=1)
            for task_name, task_logits in logits.items()
        }

    def get_probabilities(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get class probabilities (softmax of logits).

        Parameters
        ----------
        x : torch.Tensor
            Input features

        Returns
        -------
        dict
            Dictionary with class probabilities for each task
        """
        logits = self.forward(x)
        return {
            task_name: F.softmax(task_logits, dim=1)
            for task_name, task_logits in logits.items()
        }

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"MultiTaskLinearClassifier(input_dim={self.input_dim}, "
            f"task_dims={self.task_dims}, parameters={params:,})"
        )


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function that combines losses from all 4 tasks.

    Computes cross-entropy loss for each task and combines them with
    optional task-specific weights.

    Parameters
    ----------
    task_weights : dict, optional
        Weights for each task loss. If None, uses equal weights (1.0).
    """

    def __init__(self, task_weights: Dict[str, float] = None):
        super().__init__()

        self.task_names = TASK_NAMES

        # Set task weights
        if task_weights is None:
            self.task_weights = {task: 1.0 for task in self.task_names}
        else:
            self.task_weights = task_weights

        # Cross-entropy loss for each task
        self.loss_fns = nn.ModuleDict(
            {task: nn.CrossEntropyLoss() for task in self.task_names}
        )

    def forward(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss for all tasks.

        Parameters
        ----------
        outputs : dict
            Model outputs (logits) for each task
        batch : dict
            Batch dictionary containing target labels for each task

        Returns
        -------
        tuple
            (total_loss, individual_losses)
            - total_loss: weighted sum of all task losses
            - individual_losses: dict of loss for each task
        """
        individual_losses = {}
        total_loss = torch.tensor(0.0, device=outputs[self.task_names[0]].device)

        for task_name in self.task_names:
            # Get logits and targets
            logits = outputs[task_name]
            targets = batch[task_name]

            # Move targets to same device as logits
            if targets.device != logits.device:
                targets = targets.to(logits.device)

            # Compute loss
            task_loss = self.loss_fns[task_name](logits, targets)
            individual_losses[task_name] = task_loss

            # Add weighted loss
            total_loss = total_loss + self.task_weights[task_name] * task_loss

        return total_loss, individual_losses


def create_model(
    input_dim: int, task_dims: Dict[str, int]
) -> MultiTaskLinearClassifier:
    """
    Create a multi-task linear classifier.

    This is a convenience function for creating the model.

    Parameters
    ----------
    input_dim : int
        Input feature dimension (number of genes)
    task_dims : dict
        Dictionary with number of classes for each task

    Returns
    -------
    MultiTaskLinearClassifier
        The model instance
    """
    return MultiTaskLinearClassifier(input_dim=input_dim, task_dims=task_dims)

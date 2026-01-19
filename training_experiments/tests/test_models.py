"""
Tests for multi-task linear classifier model.
"""

import pytest
import torch

from training_experiments.models.linear import (
    TASK_NAMES,
    MultiTaskLinearClassifier,
    MultiTaskLoss,
    create_model,
)


class TestMultiTaskLinearClassifier:
    """Tests for the multi-task linear classifier."""

    @pytest.fixture
    def task_dims(self):
        """Sample task dimensions."""
        return {"cell_line": 10, "drug": 20, "moa_broad": 5, "moa_fine": 15}

    @pytest.fixture
    def model(self, task_dims):
        """Create a model for testing."""
        return MultiTaskLinearClassifier(input_dim=100, task_dims=task_dims)

    def test_creation(self, model, task_dims):
        """Test model creation."""
        assert model.input_dim == 100
        assert model.task_dims == task_dims
        assert len(model.task_heads) == 4

        for task_name in TASK_NAMES:
            assert task_name in model.task_heads

    def test_forward(self, model, task_dims):
        """Test forward pass."""
        batch_size = 32
        x = torch.randn(batch_size, 100)

        outputs = model(x)

        assert isinstance(outputs, dict)
        assert len(outputs) == 4

        for task_name in TASK_NAMES:
            assert task_name in outputs
            assert outputs[task_name].shape == (batch_size, task_dims[task_name])

    def test_get_predictions(self, model):
        """Test get_predictions method."""
        batch_size = 32
        x = torch.randn(batch_size, 100)

        predictions = model.get_predictions(x)

        for task_name in TASK_NAMES:
            assert task_name in predictions
            assert predictions[task_name].shape == (batch_size,)
            assert predictions[task_name].dtype == torch.long

    def test_get_probabilities(self, model, task_dims):
        """Test get_probabilities method."""
        batch_size = 32
        x = torch.randn(batch_size, 100)

        probs = model.get_probabilities(x)

        for task_name in TASK_NAMES:
            assert task_name in probs
            assert probs[task_name].shape == (batch_size, task_dims[task_name])

            # Probabilities should sum to 1
            assert torch.allclose(
                probs[task_name].sum(dim=1), torch.ones(batch_size), atol=1e-5
            )

    def test_count_parameters(self, model, task_dims):
        """Test parameter counting."""
        # Calculate expected parameters
        # For each task: input_dim * num_classes + num_classes (bias)
        expected = 0
        for task_name in TASK_NAMES:
            num_classes = task_dims[task_name]
            expected += 100 * num_classes + num_classes

        assert model.count_parameters() == expected

    def test_missing_task_raises(self):
        """Test that missing task dimension raises error."""
        task_dims = {"cell_line": 10}  # Missing other tasks

        with pytest.raises(ValueError, match="task_dims must contain"):
            MultiTaskLinearClassifier(input_dim=100, task_dims=task_dims)

    def test_repr(self, model):
        """Test string representation."""
        repr_str = repr(model)

        assert "MultiTaskLinearClassifier" in repr_str
        assert "input_dim=100" in repr_str
        assert "parameters=" in repr_str


class TestMultiTaskLoss:
    """Tests for multi-task loss function."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function."""
        return MultiTaskLoss()

    @pytest.fixture
    def task_dims(self):
        """Sample task dimensions."""
        return {"cell_line": 10, "drug": 20, "moa_broad": 5, "moa_fine": 15}

    @pytest.fixture
    def sample_batch(self, task_dims):
        """Create sample batch for testing."""
        batch_size = 32
        return {
            "X": torch.randn(batch_size, 100),
            "cell_line": torch.randint(0, task_dims["cell_line"], (batch_size,)),
            "drug": torch.randint(0, task_dims["drug"], (batch_size,)),
            "moa_broad": torch.randint(0, task_dims["moa_broad"], (batch_size,)),
            "moa_fine": torch.randint(0, task_dims["moa_fine"], (batch_size,)),
        }

    def test_forward(self, loss_fn, task_dims, sample_batch):
        """Test loss computation."""
        model = MultiTaskLinearClassifier(input_dim=100, task_dims=task_dims)
        outputs = model(sample_batch["X"])

        total_loss, individual_losses = loss_fn(outputs, sample_batch)

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.ndim == 0  # Scalar
        assert total_loss.item() >= 0

        assert len(individual_losses) == 4
        for task_name in TASK_NAMES:
            assert task_name in individual_losses
            assert individual_losses[task_name].item() >= 0

    def test_weighted_loss(self, task_dims, sample_batch):
        """Test weighted loss computation."""
        # Double weight for cell_line
        task_weights = {
            "cell_line": 2.0,
            "drug": 1.0,
            "moa_broad": 1.0,
            "moa_fine": 1.0,
        }
        loss_fn = MultiTaskLoss(task_weights=task_weights)

        model = MultiTaskLinearClassifier(input_dim=100, task_dims=task_dims)
        outputs = model(sample_batch["X"])

        total_loss, individual_losses = loss_fn(outputs, sample_batch)

        # Manually compute expected total
        expected = sum(
            task_weights[task] * individual_losses[task].item() for task in TASK_NAMES
        )

        assert total_loss.item() == pytest.approx(expected, rel=1e-5)


class TestCreateModel:
    """Tests for create_model convenience function."""

    def test_create_model(self):
        """Test model creation."""
        task_dims = {"cell_line": 10, "drug": 20, "moa_broad": 5, "moa_fine": 15}

        model = create_model(input_dim=100, task_dims=task_dims)

        assert isinstance(model, MultiTaskLinearClassifier)
        assert model.input_dim == 100


class TestGradients:
    """Tests for gradient computation."""

    def test_gradients_flow(self):
        """Test that gradients flow properly."""
        task_dims = {"cell_line": 10, "drug": 20, "moa_broad": 5, "moa_fine": 15}

        model = MultiTaskLinearClassifier(input_dim=100, task_dims=task_dims)
        loss_fn = MultiTaskLoss()

        batch_size = 32
        batch = {
            "X": torch.randn(batch_size, 100),
            "cell_line": torch.randint(0, 10, (batch_size,)),
            "drug": torch.randint(0, 20, (batch_size,)),
            "moa_broad": torch.randint(0, 5, (batch_size,)),
            "moa_fine": torch.randint(0, 15, (batch_size,)),
        }

        outputs = model(batch["X"])
        total_loss, _ = loss_fn(outputs, batch)

        # Compute gradients
        total_loss.backward()

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

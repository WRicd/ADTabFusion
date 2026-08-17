"""GPU-accelerated deep tabular classifiers with a scikit-learn interface.

Both estimators expect the already-preprocessed dense numeric matrix produced
by :func:`src.preprocessing.build_preprocessor` (imputed, scaled, one-hot
encoded), so they can be dropped into the existing ``Pipeline`` unchanged.

They implement ``fit`` / ``predict`` / ``predict_proba`` and accept
``sample_weight``, which the Phase D transition model relies on for
subject-balanced pair weighting.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight

LOGGER = logging.getLogger(__name__)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "PyTorch is required for deep tabular models. "
            "Install it from https://pytorch.org (CUDA build recommended)."
        ) from exc
    return torch


def resolve_device(requested: str) -> str:
    """Return a usable torch device string, falling back to CPU."""
    torch = _require_torch()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested


def _ft_transformer_net_class():
    """Build the FT-Transformer ``nn.Module`` class, once, at module scope.

    torch is imported lazily so this module stays importable without it, but the
    class must still be reachable as ``deep_tabular.FTTransformerNet`` -- pickle
    resolves classes by qualified name and cannot serialize instances of classes
    defined inside a function. Defining it locally silently corrupted
    ``joblib.dump`` output, which is what the project's model manifests hash.
    """
    cached = globals().get("FTTransformerNet")
    if cached is not None:
        return cached

    torch = _require_torch()

    class FTTransformerNet(torch.nn.Module):
        def __init__(
            self,
            n_features: int,
            n_classes: int,
            d_token: int,
            n_heads: int,
            n_blocks: int,
            ffn_factor: float,
            attention_dropout: float,
            ffn_dropout: float,
        ):
            super().__init__()
            # Per-feature linear tokenizer: token_i = x_i * W_i + b_i
            self.feature_weight = torch.nn.Parameter(torch.empty(n_features, d_token))
            self.feature_bias = torch.nn.Parameter(torch.empty(n_features, d_token))
            self.cls_token = torch.nn.Parameter(torch.empty(1, 1, d_token))
            torch.nn.init.normal_(self.feature_weight, std=d_token**-0.5)
            torch.nn.init.normal_(self.feature_bias, std=d_token**-0.5)
            torch.nn.init.normal_(self.cls_token, std=d_token**-0.5)

            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=n_heads,
                dim_feedforward=int(d_token * ffn_factor),
                dropout=attention_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_blocks, enable_nested_tensor=False)
            self.head = torch.nn.Sequential(
                torch.nn.LayerNorm(d_token),
                torch.nn.GELU(),
                torch.nn.Dropout(ffn_dropout),
                torch.nn.Linear(d_token, n_classes),
            )

        def forward(self, x):
            # x: (batch, n_features) -> tokens: (batch, n_features, d_token)
            tokens = x.unsqueeze(-1) * self.feature_weight + self.feature_bias
            cls = self.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            encoded = self.encoder(tokens)
            return self.head(encoded[:, 0])

    # pickle looks the class up as <__module__>.<__qualname__>, so both must
    # point at the module-level alias registered below.
    FTTransformerNet.__module__ = __name__
    FTTransformerNet.__qualname__ = "FTTransformerNet"
    globals()["FTTransformerNet"] = FTTransformerNet
    return FTTransformerNet


class _BaseTorchClassifier(ClassifierMixin, BaseEstimator):
    """Shared training loop for the torch-backed tabular classifiers."""

    def _build_network(self, n_features: int, n_classes: int):
        raise NotImplementedError

    # -- sklearn plumbing ---------------------------------------------------
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = False
        return tags

    # -- serialization ------------------------------------------------------
    def __getstate__(self):
        """Serialize with CPU tensors so artifacts load on any machine.

        A CUDA-trained network pickles its tensors with device metadata and
        then fails to load where no GPU exists. Frozen model artifacts have to
        outlive the machine that produced them, so move to CPU on the way out.
        """
        state = dict(self.__dict__)
        network = state.get("network_")
        if network is not None:
            state["network_"] = copy.deepcopy(network).cpu()
            state["device_"] = "cpu"
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        network = self.__dict__.get("network_")
        if network is None:
            return
        # Restore onto the best device actually available at load time.
        device = resolve_device(self.__dict__.get("device", "cpu"))
        self.__dict__["network_"] = network.to(device)
        self.__dict__["device_"] = device

    # -- training -----------------------------------------------------------
    def fit(self, X, y, sample_weight=None, eval_set=None):
        torch = _require_torch()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        n_classes = len(self.classes_)

        class_index = {label: index for index, label in enumerate(self.classes_)}
        y_index = np.asarray([class_index[value] for value in y], dtype=np.int64)

        device = resolve_device(self.device)
        self.device_ = device
        torch.manual_seed(self.random_state)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(self.random_state)

        generator = torch.Generator().manual_seed(self.random_state)
        self.network_ = self._build_network(self.n_features_in_, n_classes).to(device)

        # Class weighting mirrors sklearn's ``class_weight="balanced"``.
        if self.class_weight == "balanced":
            weights = compute_class_weight("balanced", classes=self.classes_, y=y)
            class_weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        else:
            class_weight_tensor = None

        criterion = torch.nn.CrossEntropyLoss(weight=class_weight_tensor, reduction="none")
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        X_tensor = torch.tensor(X, device=device)
        y_tensor = torch.tensor(y_index, device=device)
        if sample_weight is None:
            weight_tensor = torch.ones(len(y_index), dtype=torch.float32, device=device)
        else:
            weight_tensor = torch.tensor(np.asarray(sample_weight, dtype=np.float32), device=device)

        eval_tensors = None
        if eval_set is not None and self.early_stopping:
            X_val, y_val = eval_set
            y_val_index = np.asarray([class_index[value] for value in np.asarray(y_val)], dtype=np.int64)
            eval_tensors = (
                torch.tensor(np.asarray(X_val, dtype=np.float32), device=device),
                torch.tensor(y_val_index, device=device),
            )

        n_samples = len(y_index)
        batch_size = min(self.batch_size, n_samples)
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        patience_left = self.patience
        self.n_epochs_run_ = 0

        for epoch in range(self.max_epochs):
            self.network_.train()
            permutation = torch.randperm(n_samples, generator=generator).to(device)
            for start in range(0, n_samples, batch_size):
                index = permutation[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = self.network_(X_tensor[index])
                losses = criterion(logits, y_tensor[index])
                loss = (losses * weight_tensor[index]).sum() / weight_tensor[index].sum()
                loss.backward()
                optimizer.step()
            self.n_epochs_run_ = epoch + 1

            if eval_tensors is None:
                continue

            self.network_.eval()
            with torch.no_grad():
                val_logits = self.network_(eval_tensors[0])
                val_loss = float(
                    torch.nn.functional.cross_entropy(val_logits, eval_tensors[1], weight=class_weight_tensor).item()
                )
            if val_loss < best_loss - self.tol:
                best_loss = val_loss
                best_state = {key: value.detach().clone() for key, value in self.network_.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    LOGGER.info("Early stopping at epoch %d (val loss %.4f).", epoch + 1, best_loss)
                    break

        if best_state is not None:
            self.network_.load_state_dict(best_state)
            self.best_validation_loss_ = best_loss

        return self

    # -- inference ----------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        torch = _require_torch()
        if not hasattr(self, "network_"):
            raise RuntimeError("Estimator is not fitted; call fit() first.")

        X = np.asarray(X, dtype=np.float32)
        self.network_.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), self.inference_batch_size):
                chunk = torch.tensor(X[start : start + self.inference_batch_size], device=self.device_)
                logits = self.network_(chunk)
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        if not outputs:
            return np.empty((0, len(self.classes_)), dtype=np.float64)
        proba = np.vstack(outputs).astype(np.float64)
        # torch computes softmax in float32, which leaves rows ~1e-7 off the
        # simplex. sklearn's log_loss / brier_score use a far tighter
        # tolerance, and Phase D feeds these probabilities into calibration,
        # so renormalize once in float64.
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class TabularMLPClassifier(_BaseTorchClassifier):
    """Residual MLP for tabular data, trained on GPU when available.

    Parameters mirror the surrounding project's config style; every argument
    is stored unmodified so ``sklearn.clone`` and ``get_params`` behave.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] | list[int] = (256, 128),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        inference_batch_size: int = 4096,
        max_epochs: int = 200,
        patience: int = 20,
        tol: float = 1e-4,
        class_weight: str | None = "balanced",
        early_stopping: bool = True,
        device: str = "cpu",
        random_state: int = 42,
    ):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.tol = tol
        self.class_weight = class_weight
        self.early_stopping = early_stopping
        self.device = device
        self.random_state = random_state

    def _build_network(self, n_features: int, n_classes: int):
        torch = _require_torch()

        layers: list[Any] = []
        input_size = n_features
        for hidden in self.hidden_sizes:
            layers.extend(
                [
                    torch.nn.Linear(input_size, hidden),
                    torch.nn.BatchNorm1d(hidden),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(self.dropout),
                ]
            )
            input_size = hidden
        layers.append(torch.nn.Linear(input_size, n_classes))
        return torch.nn.Sequential(*layers)


class FTTransformerClassifier(_BaseTorchClassifier):
    """Feature-Tokenizer Transformer (Gorishniy et al., 2021) for tabular data.

    Every preprocessed column is treated as a numerical feature and projected
    to its own ``d_token``-dimensional embedding; a CLS token is prepended and
    the transformer's CLS output feeds the classification head.
    """

    def __init__(
        self,
        d_token: int = 64,
        n_blocks: int = 3,
        n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        ffn_factor: float = 2.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 128,
        inference_batch_size: int = 1024,
        max_epochs: int = 150,
        patience: int = 16,
        tol: float = 1e-4,
        class_weight: str | None = "balanced",
        early_stopping: bool = True,
        device: str = "cpu",
        random_state: int = 42,
    ):
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.ffn_factor = ffn_factor
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.tol = tol
        self.class_weight = class_weight
        self.early_stopping = early_stopping
        self.device = device
        self.random_state = random_state

    def _build_network(self, n_features: int, n_classes: int):
        d_token = self.d_token
        # nn.MultiheadAttention requires d_token to be divisible by n_heads.
        n_heads = self.n_heads
        while n_heads > 1 and d_token % n_heads != 0:
            n_heads -= 1

        return _ft_transformer_net_class()(
            n_features=n_features,
            n_classes=n_classes,
            d_token=d_token,
            n_heads=n_heads,
            n_blocks=self.n_blocks,
            ffn_factor=self.ffn_factor,
            attention_dropout=self.attention_dropout,
            ffn_dropout=self.ffn_dropout,
        )

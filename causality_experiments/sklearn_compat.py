from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression as _SklearnLogisticRegression
    from sklearn.preprocessing import StandardScaler as _SklearnStandardScaler
except ModuleNotFoundError:  # pragma: no cover - exercised indirectly in this repo
    _SklearnLogisticRegression = None
    _SklearnStandardScaler = None


if _SklearnStandardScaler is not None:
    StandardScaler = _SklearnStandardScaler
else:

    class StandardScaler:
        def fit(self, x: np.ndarray) -> "StandardScaler":
            x = np.asarray(x, dtype=np.float64)
            self.mean_ = x.mean(axis=0)
            scale = x.std(axis=0)
            scale[scale == 0.0] = 1.0
            self.scale_ = scale
            return self

        def transform(self, x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            return (x - self.mean_) / self.scale_

        def fit_transform(self, x: np.ndarray) -> np.ndarray:
            return self.fit(x).transform(x)


if _SklearnLogisticRegression is not None:
    LogisticRegression = _SklearnLogisticRegression
else:

    class LogisticRegression:
        def __init__(
            self,
            *,
            penalty: str = "l1",
            solver: str = "liblinear",
            C: float = 1.0,
            max_iter: int = 1000,
            random_state: int | None = None,
            class_weight: dict[int, float] | None = None,
        ) -> None:
            if penalty != "l1":
                raise ValueError("This local LogisticRegression fallback only supports penalty='l1'.")
            if solver != "liblinear":
                raise ValueError("This local LogisticRegression fallback only supports solver='liblinear'.")
            self.C = float(C)
            self.max_iter = int(max_iter)
            self.random_state = random_state
            self.class_weight = class_weight

        def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticRegression":
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.int64)
            classes = np.unique(y)
            if len(classes) != 2:
                raise NotImplementedError("This local LogisticRegression fallback currently supports binary classification only.")
            self.classes_ = classes
            positive = classes[1]
            y_bin = (y == positive).astype(np.float64)
            sample_weight = np.ones(len(y_bin), dtype=np.float64)
            if self.class_weight:
                for label, value in self.class_weight.items():
                    sample_weight[y == label] = float(value)
            weight_sum = float(sample_weight.sum())
            l1_strength = 1.0 / max(self.C * weight_sum, 1e-12)
            lipschitz = 0.25 * float(np.linalg.norm(x, ord=2) ** 2) / max(len(x), 1)
            step = 1.0 / max(lipschitz, 1e-6)

            def _sigmoid(values: np.ndarray) -> np.ndarray:
                clipped = np.clip(values, -50.0, 50.0)
                return 1.0 / (1.0 + np.exp(-clipped))

            def _soft_threshold(values: np.ndarray, thresh: float) -> np.ndarray:
                return np.sign(values) * np.maximum(np.abs(values) - thresh, 0.0)

            w = np.zeros(x.shape[1], dtype=np.float64)
            b = 0.0
            z_w = w.copy()
            z_b = b
            t = 1.0
            for _ in range(self.max_iter):
                logits = x @ z_w + z_b
                pred = _sigmoid(logits)
                error = (pred - y_bin) * sample_weight
                grad_w = x.T @ error / weight_sum
                grad_b = float(error.sum() / weight_sum)
                next_w = _soft_threshold(z_w - step * grad_w, step * l1_strength)
                next_b = z_b - step * grad_b
                next_t = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
                z_w = next_w + ((t - 1.0) / next_t) * (next_w - w)
                z_b = next_b + ((t - 1.0) / next_t) * (next_b - b)
                w = next_w
                b = next_b
                t = next_t
            self.coef_ = w[None, :]
            self.intercept_ = np.array([b], dtype=np.float64)
            return self

        def decision_function(self, x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            return x @ self.coef_[0] + self.intercept_[0]

        def predict(self, x: np.ndarray) -> np.ndarray:
            score = self.decision_function(x)
            return np.where(score > 0.0, self.classes_[1], self.classes_[0])


from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .data import DatasetBundle


def _safe_abs_corr(values: torch.Tensor, target: torch.Tensor) -> float:
    x = values.float()
    y = target.float()
    if x.numel() == 0 or y.numel() == 0:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(torch.sum(x**2) * torch.sum(y**2)).clamp_min(1e-12)
    return float(torch.abs(torch.sum(x * y) / denom).item())


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-12:
        return 0.0
    return numerator / denominator


def build_feature_clue_rows(bundle: DatasetBundle, split_name: str = "train") -> list[dict[str, Any]]:
    split = bundle.split(split_name)
    x = split["x"]
    y = split["y"]
    env = split["env"]
    metadata = bundle.metadata or {}
    supervision_source = str(metadata.get("causal_supervision", "none"))
    feature_names = list(metadata.get("feature_columns", []))
    if len(feature_names) != x.shape[1]:
        feature_names = [f"feature_{index}" for index in range(x.shape[1])]

    causal_mask = bundle.causal_mask
    cause_position = metadata.get("cause_position")
    feature_count = max(int(x.shape[1]), 1)
    rows: list[dict[str, Any]] = []
    for index, feature_name in enumerate(feature_names):
        values = x[:, index]
        label_corr = _safe_abs_corr(values, y)
        env_corr = _safe_abs_corr(values, env)
        corr_margin = label_corr - env_corr
        row: dict[str, Any] = {
            "dataset": bundle.name,
            "split": split_name,
            "feature_index": index,
            "feature_index_frac": float(index / max(feature_count - 1, 1)),
            "feature_name": feature_name,
            "label_corr": label_corr,
            "env_corr": env_corr,
            "corr_margin": corr_margin,
            "abs_corr_margin": abs(corr_margin),
            "corr_sum": label_corr + env_corr,
            "label_env_ratio": _safe_ratio(label_corr, env_corr),
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
            "abs_mean": float(values.abs().mean().item()),
            "has_ground_truth_mask": causal_mask is not None,
            "has_explicit_supervision": supervision_source == "explicit_mask",
            "supervision_source": supervision_source,
            "supervision_explicit": 1.0 if supervision_source == "explicit_mask" else 0.0,
            "supervision_derived": 1.0 if supervision_source == "derived_mask" else 0.0,
            "supervision_none": 1.0 if supervision_source == "none" else 0.0,
            "causal_target": float(causal_mask[index].item()) if causal_mask is not None else float("nan"),
            "has_cause_position": cause_position is not None,
            "cause_position_target": 1.0 if cause_position == index else 0.0,
            "task": bundle.task,
            "modality": str(metadata.get("modality", "")),
            "modality_features": 1.0 if str(metadata.get("modality", "")) == "features" else 0.0,
            "modality_sequence": 1.0 if str(metadata.get("modality", "")) == "sequence" else 0.0,
            "task_classification": 1.0 if bundle.task == "classification" else 0.0,
            "utility_target": float("nan"),
            "utility_weight": 0.0,
            "utility_value": float("nan"),
            "utility_count": 0.0,
        }
        row["soft_causal_target"] = aggregate_soft_causal_target(row)
        rows.append(row)
    return rows


def aggregate_soft_causal_target(row: dict[str, Any]) -> float:
    if bool(row.get("has_explicit_supervision", False)):
        return float(row.get("causal_target", 0.0))
    if bool(row.get("has_cause_position", False)):
        return float(row.get("cause_position_target", 0.0))
    margin = float(row.get("corr_margin", 0.0))
    return float(torch.sigmoid(torch.tensor(margin * 6.0)).item())


def aggregate_rank_target(row: dict[str, Any], *, utility_blend: float = 0.5) -> float:
    utility_target = row.get("utility_target")
    utility_weight = float(row.get("utility_weight", 0.0) or 0.0)
    base_target = aggregate_soft_causal_target(row)
    if utility_target is None:
        return base_target
    utility_value = float(utility_target)
    if not torch.isfinite(torch.tensor(utility_value)) or utility_weight <= 0.0:
        return base_target
    blend = min(max(utility_blend, 0.0), 1.0) * min(max(utility_weight, 0.0), 1.0)
    return (1.0 - blend) * base_target + blend * utility_value


def _feature_value(row: dict[str, Any], key: str) -> float:
    if key in row:
        value = row[key]
        if isinstance(value, bool):
            return float(value)
        if value in (None, ""):
            return 0.0
        return float(value)
    if key == "abs_corr_margin":
        return abs(float(row.get("corr_margin", 0.0) or 0.0))
    if key == "corr_sum":
        return float(row.get("label_corr", 0.0) or 0.0) + float(row.get("env_corr", 0.0) or 0.0)
    if key == "label_env_ratio":
        label_corr = float(row.get("label_corr", 0.0) or 0.0)
        env_corr = float(row.get("env_corr", 0.0) or 0.0)
        return _safe_ratio(label_corr, env_corr)
    if key == "abs_mean":
        return abs(float(row.get("mean", 0.0) or 0.0))
    if key == "feature_index_frac":
        return 0.0
    if key == "supervision_explicit":
        return 1.0 if str(row.get("supervision_source", "")).lower() == "explicit_mask" else 0.0
    if key == "supervision_derived":
        return 1.0 if str(row.get("supervision_source", "")).lower() == "derived_mask" else 0.0
    if key == "supervision_none":
        source = str(row.get("supervision_source", "none")).lower()
        return 1.0 if source in {"", "none"} else 0.0
    if key == "modality_features":
        return 1.0 if str(row.get("modality", "")).lower() == "features" else 0.0
    if key == "modality_sequence":
        return 1.0 if str(row.get("modality", "")).lower() == "sequence" else 0.0
    if key == "task_classification":
        return 1.0 if str(row.get("task", "")).lower() == "classification" else 0.0
    return 0.0


def clue_feature_vector(
    row: dict[str, Any],
    feature_columns: list[str] | tuple[str, ...] | None = None,
) -> list[float]:
    columns = DISCOVERY_FEATURE_COLUMNS if feature_columns is None else feature_columns
    return [_feature_value(row, key) for key in columns]


DISCOVERY_FEATURE_COLUMNS = (
    "label_corr",
    "env_corr",
    "corr_margin",
    "abs_corr_margin",
    "corr_sum",
    "label_env_ratio",
    "mean",
    "abs_mean",
    "std",
    "feature_index_frac",
    "supervision_explicit",
    "supervision_derived",
    "supervision_none",
    "modality_features",
    "modality_sequence",
    "task_classification",
)


class DiscoveryScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rank_head = nn.Linear(hidden_dim, 1)
        self.support_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.rank_head(hidden), self.support_head(hidden)


def build_discovery_model(input_dim: int, hidden_dim: int = 32) -> DiscoveryScorer:
    return DiscoveryScorer(input_dim=input_dim, hidden_dim=hidden_dim)


def combine_discovery_scores(rank_logits: torch.Tensor, support_logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(rank_logits) * torch.sigmoid(support_logits)


def clue_tensor(rows: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor(
        [clue_feature_vector(row) for row in rows],
        dtype=torch.float32,
    )
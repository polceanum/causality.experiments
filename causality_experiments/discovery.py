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
        [[float(row[key]) for key in DISCOVERY_FEATURE_COLUMNS] for row in rows],
        dtype=torch.float32,
    )
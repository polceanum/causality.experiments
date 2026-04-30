from __future__ import annotations

from collections.abc import Mapping, Sequence
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


LANGUAGE_CLUE_FEATURE_COLUMNS = (
    "language_causal_score",
    "language_spurious_score",
    "language_ambiguous_score",
    "language_confidence",
    "language_source_count",
)

IMAGE_CLUE_FEATURE_COLUMNS = (
    "image_label_score",
    "image_background_score",
    "image_group_stability",
    "image_prompt_margin",
    "image_confidence",
    "prototype_source_count",
)

BRIDGE_CLUE_FEATURE_COLUMNS = (
    "top_activation_group_entropy",
    "label_env_disentanglement",
    "clue_source_count",
)

DISCOVERY_CLUE_AUDIT_COLUMNS = (
    "language_prior_source",
    "feature_card_path",
    "clue_source_mask",
)

EXTERNAL_CLUE_COLUMNS = (
    *LANGUAGE_CLUE_FEATURE_COLUMNS,
    *IMAGE_CLUE_FEATURE_COLUMNS,
    *BRIDGE_CLUE_FEATURE_COLUMNS,
    *DISCOVERY_CLUE_AUDIT_COLUMNS,
)


def _default_external_clue_values() -> dict[str, Any]:
    values: dict[str, Any] = {
        key: 0.0
        for key in (
            *LANGUAGE_CLUE_FEATURE_COLUMNS,
            *IMAGE_CLUE_FEATURE_COLUMNS,
            *BRIDGE_CLUE_FEATURE_COLUMNS,
        )
    }
    values.update({key: "" for key in DISCOVERY_CLUE_AUDIT_COLUMNS})
    return values


def build_feature_clue_rows(
    bundle: DatasetBundle,
    split_name: str = "train",
    external_clues: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
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
            **_default_external_clue_values(),
        }
        row["label_env_disentanglement"] = abs(corr_margin)
        row["soft_causal_target"] = aggregate_soft_causal_target(row)
        rows.append(row)
    if external_clues:
        rows = merge_external_clue_rows(rows, external_clues)
    return rows


def _clue_key(row: Mapping[str, Any]) -> tuple[str, str] | None:
    feature_name = str(row.get("feature_name", "")).strip()
    if not feature_name:
        return None
    return str(row.get("dataset", "")).strip(), feature_name


def _clue_float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not torch.isfinite(torch.tensor(parsed)):
        return 0.0
    return parsed


def _source_mask(row: Mapping[str, Any]) -> str:
    sources = []
    if _clue_float(row, "language_confidence") > 0.0:
        sources.append("language")
    if _clue_float(row, "image_confidence") > 0.0:
        sources.append("image")
    if _clue_float(row, "top_activation_group_entropy") > 0.0:
        sources.append("bridge")
    return ",".join(sources)


def merge_external_clue_rows(
    feature_rows: Sequence[Mapping[str, Any]],
    external_clue_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    exact: dict[tuple[str, str], dict[str, Any]] = {}
    by_feature: dict[str, dict[str, Any]] = {}
    for clue in external_clue_rows:
        key = _clue_key(clue)
        if key is None:
            continue
        dataset, feature_name = key
        target: dict[str, dict[str, Any]]
        target_key: tuple[str, str] | str
        if dataset:
            target = exact
            target_key = (dataset, feature_name)
        else:
            target = by_feature
            target_key = feature_name
        merged = dict(target.get(target_key, {}))
        merged.update(dict(clue))
        target[target_key] = merged

    key_columns = {"dataset", "split", "feature_index", "feature_name"}
    merged_rows: list[dict[str, Any]] = []
    for row in feature_rows:
        updated = dict(row)
        for key, value in _default_external_clue_values().items():
            updated.setdefault(key, value)
        feature_key = _clue_key(updated)
        clue = None
        if feature_key is not None:
            dataset, feature_name = feature_key
            clue = exact.get((dataset, feature_name)) or by_feature.get(feature_name)
        if clue is not None:
            for key, value in clue.items():
                if key in key_columns:
                    continue
                updated[key] = value
        source_mask = _source_mask(updated)
        updated["clue_source_mask"] = source_mask
        updated["clue_source_count"] = float(len([part for part in source_mask.split(",") if part]))
        merged_rows.append(updated)
    return merged_rows


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

DISCOVERY_FEATURE_COLUMNS_V2 = (
    *DISCOVERY_FEATURE_COLUMNS,
    *LANGUAGE_CLUE_FEATURE_COLUMNS,
    *IMAGE_CLUE_FEATURE_COLUMNS,
    *BRIDGE_CLUE_FEATURE_COLUMNS,
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
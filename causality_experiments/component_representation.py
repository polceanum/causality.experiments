from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

import pandas as pd
import torch

from .data import DatasetBundle
from .discovery import build_feature_clue_rows


COMPONENT_REPRESENTATION_SCHEMA_VERSION = "component_representation/v1"
COMPONENT_CLUE_SCHEMA_VERSION = "component_clue/v1"
ADAPTER_SCHEMA_VERSION = "component_adapter/v1"

_COMPONENT_FEATURE_RE = re.compile(r"^feature_(?P<component>[A-Za-z][A-Za-z0-9_]*)_\d+$")


@dataclass(frozen=True)
class ComponentAdapterResult:
    feature_frame: pd.DataFrame
    report: dict[str, Any]


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_abs_corr(values: torch.Tensor, target: torch.Tensor) -> float:
    values = values.detach().float().view(-1)
    target = target.detach().float().view(-1)
    if values.numel() < 2 or target.numel() < 2:
        return 0.0
    values = values - values.mean()
    target = target - target.mean()
    denominator = values.norm() * target.norm()
    if float(denominator.item()) <= 1e-12:
        return 0.0
    return float(torch.abs(torch.dot(values, target) / denominator).item())


def _canonical_component_name(name: str) -> str:
    value = name.strip().lower().replace("-", "_").replace(" ", "_")
    return value or "global"


def feature_columns_from_frame(frame: pd.DataFrame) -> list[str]:
    feature_columns = [col for col in frame.columns if col.startswith("feature_") or col.startswith("x")]
    if feature_columns:
        return feature_columns
    ignored = {"split", "fold", "y", "label", "target", "bird_label", "place", "background", "env", "spurious", "group", "group_id"}
    return [col for col in frame.columns if col not in ignored and pd.api.types.is_numeric_dtype(frame[col])]


def load_feature_components_from_manifest(manifest_path: Path) -> dict[str, list[str]]:
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        _canonical_component_name(str(key)): [str(item) for item in value]
        for key, value in dict(payload.get("feature_components", {})).items()
    }


def infer_feature_components(
    feature_columns: Sequence[str],
    manifest_components: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, list[str]]:
    feature_set = {str(column) for column in feature_columns}
    groups: dict[str, list[str]] = {}
    for component, columns in dict(manifest_components or {}).items():
        retained = [str(column) for column in columns if str(column) in feature_set]
        if retained:
            groups[_canonical_component_name(str(component))] = retained
    if groups:
        assigned = {column for columns in groups.values() for column in columns}
        leftover = [str(column) for column in feature_columns if str(column) not in assigned]
        if leftover:
            groups.setdefault("global", []).extend(leftover)
        return groups

    for column in feature_columns:
        match = _COMPONENT_FEATURE_RE.match(str(column))
        component = _canonical_component_name(match.group("component")) if match else "global"
        groups.setdefault(component, []).append(str(column))
    return groups or {"global": [str(column) for column in feature_columns]}


def feature_component_lookup(feature_components: Mapping[str, Sequence[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for component, columns in feature_components.items():
        for column in columns:
            lookup[str(column)] = _canonical_component_name(str(component))
    return lookup


def filter_feature_components(
    feature_components: Mapping[str, Sequence[str]],
    *,
    include_components: Sequence[str] = (),
    exclude_components: Sequence[str] = (),
) -> dict[str, list[str]]:
    include = {_canonical_component_name(component) for component in include_components if str(component).strip()}
    exclude = {_canonical_component_name(component) for component in exclude_components if str(component).strip()}
    filtered: dict[str, list[str]] = {}
    for component, columns in feature_components.items():
        key = _canonical_component_name(str(component))
        if include and key not in include:
            continue
        if key in exclude:
            continue
        filtered[key] = [str(column) for column in columns]
    if not filtered:
        raise ValueError("Component filter removed all feature columns.")
    return filtered


def ordered_feature_components(feature_columns: Sequence[str], component_names: Sequence[str]) -> dict[str, list[str]]:
    names = [_canonical_component_name(str(component)) for component in component_names if str(component).strip()]
    columns = [str(column) for column in feature_columns]
    if not names:
        return infer_feature_components(columns)
    if len(columns) % len(names) != 0:
        raise ValueError("Ordered component names require the feature width to be divisible by the component count.")
    component_dim = len(columns) // len(names)
    return {
        component: columns[index * component_dim : (index + 1) * component_dim]
        for index, component in enumerate(names)
    }


def columns_for_components(feature_components: Mapping[str, Sequence[str]]) -> list[str]:
    columns: list[str] = []
    for component_columns in feature_components.values():
        columns.extend(str(column) for column in component_columns)
    return columns


def component_summary_rows(
    bundle: DatasetBundle,
    *,
    feature_components: Mapping[str, Sequence[str]] | None = None,
    split_name: str = "train",
) -> list[dict[str, Any]]:
    metadata = bundle.metadata or {}
    feature_columns = [str(column) for column in metadata.get("feature_columns", [])]
    if len(feature_columns) != bundle.input_dim:
        feature_columns = [f"feature_{index}" for index in range(bundle.input_dim)]
    components = infer_feature_components(feature_columns, feature_components)
    split = bundle.split(split_name)
    rows: list[dict[str, Any]] = []
    for component, columns in components.items():
        indices = [feature_columns.index(column) for column in columns if column in feature_columns]
        if not indices:
            continue
        values = split["x"][:, indices].mean(dim=1)
        label_corr = _safe_abs_corr(values, split["y"])
        env_corr = _safe_abs_corr(values, split["env"])
        corr_margin = label_corr - env_corr
        rows.append(
            {
                "schema_version": COMPONENT_CLUE_SCHEMA_VERSION,
                "dataset": bundle.name,
                "split": split_name,
                "component_group": component,
                "component_feature_count": len(indices),
                "component_label_corr": label_corr,
                "component_env_corr": env_corr,
                "component_corr_margin": corr_margin,
                "component_causal_score": max(corr_margin, 0.0),
                "component_shortcut_score": max(env_corr - label_corr, 0.0),
                "feature_names": ",".join(columns),
            }
        )
    rows.sort(key=lambda row: (float(row["component_causal_score"]), str(row["component_group"])), reverse=True)
    return rows


def build_component_clue_rows(
    bundle: DatasetBundle,
    *,
    feature_components: Mapping[str, Sequence[str]] | None = None,
    split_name: str = "train",
) -> list[dict[str, Any]]:
    feature_rows = build_feature_clue_rows(bundle, split_name=split_name)
    feature_columns = [str(row["feature_name"]) for row in feature_rows]
    components = infer_feature_components(feature_columns, feature_components)
    component_by_feature = feature_component_lookup(components)
    summaries = {
        str(row["component_group"]): row
        for row in component_summary_rows(bundle, feature_components=components, split_name=split_name)
    }
    rows: list[dict[str, Any]] = []
    for row in feature_rows:
        updated = dict(row)
        component = component_by_feature.get(str(row.get("feature_name", "")), "global")
        summary = summaries.get(component, {})
        label_corr = _safe_float(updated.get("label_corr"))
        env_corr = _safe_float(updated.get("env_corr"))
        corr_margin = label_corr - env_corr
        causal_score = max(corr_margin, 0.0)
        shortcut_score = max(env_corr - label_corr, 0.0)
        updated.update(
            {
                "schema_version": COMPONENT_CLUE_SCHEMA_VERSION,
                "component_group": component,
                "component_feature_count": _safe_float(summary.get("component_feature_count")),
                "component_label_corr": _safe_float(summary.get("component_label_corr")),
                "component_env_corr": _safe_float(summary.get("component_env_corr")),
                "component_corr_margin": _safe_float(summary.get("component_corr_margin")),
                "feature_causal_score": causal_score,
                "feature_shortcut_score": shortcut_score,
                "component_causal_score": _safe_float(summary.get("component_causal_score")),
                "component_shortcut_score": _safe_float(summary.get("component_shortcut_score")),
                "adapter_prior": min(max(0.5 + 0.5 * corr_margin, 0.0), 1.0),
            }
        )
        rows.append(updated)
    return rows


def component_test_rows(
    bundle: DatasetBundle,
    *,
    feature_components: Mapping[str, Sequence[str]] | None = None,
    split_name: str = "train",
) -> list[dict[str, Any]]:
    rows = component_summary_rows(bundle, feature_components=feature_components, split_name=split_name)
    if not rows:
        return []
    control_by_component: dict[str, dict[str, Any]] = {}
    ordered = sorted(rows, key=lambda row: str(row["component_group"]))
    for index, row in enumerate(ordered):
        control_by_component[str(row["component_group"])] = ordered[(index + 1) % len(ordered)]
    tested: list[dict[str, Any]] = []
    for row in rows:
        control = control_by_component[str(row["component_group"])]
        label_delta = float(row["component_label_corr"])
        env_delta = float(row["component_env_corr"])
        random_delta = float(control["component_label_corr"])
        effect_margin = label_delta - env_delta - random_delta
        tested.append(
            {
                **row,
                "action": "component_signal_check",
                "control_component_group": control["component_group"],
                "test_effect_label_delta": label_delta,
                "test_effect_env_delta": env_delta,
                "test_random_control_delta": random_delta,
                "test_effect_selectivity": label_delta - env_delta,
                "test_passed_control": effect_margin > 0.0 and label_delta >= random_delta,
                "test_cost": 1.0,
            }
        )
    return tested


def adapter_priors_from_clues(
    feature_columns: Sequence[str],
    clue_rows: Sequence[Mapping[str, Any]] | None = None,
) -> torch.Tensor:
    priors = {str(column): 0.5 for column in feature_columns}
    for row in clue_rows or []:
        feature_name = str(row.get("feature_name", ""))
        if feature_name not in priors:
            continue
        label_corr = _safe_float(row.get("label_corr"))
        env_corr = _safe_float(row.get("env_corr"))
        prior = row.get("adapter_prior")
        if prior not in (None, ""):
            priors[feature_name] = min(max(_safe_float(prior, default=0.5), 0.0), 1.0)
        else:
            priors[feature_name] = min(max(0.5 + 0.5 * (label_corr - env_corr), 0.0), 1.0)
    return torch.tensor([priors[str(column)] for column in feature_columns], dtype=torch.float32)


def _metadata_columns(frame: pd.DataFrame, feature_columns: Sequence[str]) -> list[str]:
    feature_set = set(feature_columns)
    return [column for column in frame.columns if column not in feature_set]


def _group_balanced_weights(frame: pd.DataFrame) -> torch.Tensor:
    if "group" not in frame.columns:
        return torch.ones(len(frame), dtype=torch.float32)
    groups = frame["group"].astype(int).tolist()
    counts = {group: max(groups.count(group), 1) for group in set(groups)}
    weights = torch.tensor([1.0 / counts[group] for group in groups], dtype=torch.float32)
    return weights / weights.mean().clamp_min(1e-12)


def _env_covariance_penalty(adapted_x: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
    centered_x = adapted_x - adapted_x.mean(dim=0, keepdim=True)
    centered_env = env.float() - env.float().mean()
    denom = centered_x.pow(2).mean(dim=0).sqrt().clamp_min(1e-6) * centered_env.pow(2).mean().sqrt().clamp_min(1e-6)
    corr = (centered_x * centered_env[:, None]).mean(dim=0) / denom
    return corr.abs().mean()


def train_component_adapter(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    clue_rows: Sequence[Mapping[str, Any]] | None = None,
    epochs: int = 200,
    lr: float = 0.05,
    env_penalty_weight: float = 0.2,
    env_adversary_weight: float = 0.0,
    clue_prior_weight: float = 1.0,
    output_prefix: str = "feature_adapted",
    seed: int = 0,
) -> ComponentAdapterResult:
    if "split" not in frame.columns:
        raise ValueError("Component adapter input requires a split column.")
    label_col = next((col for col in ("y", "label", "target", "bird_label") if col in frame.columns), None)
    env_col = next((col for col in ("place", "background", "env", "spurious") if col in frame.columns), None)
    if label_col is None or env_col is None:
        raise ValueError("Component adapter input requires label and environment columns.")
    columns = [str(column) for column in (feature_columns or feature_columns_from_frame(frame))]
    if not columns:
        raise ValueError("Component adapter input has no feature columns.")

    train_mask = frame["split"].astype(str).str.lower() == "train"
    if not bool(train_mask.any()):
        raise ValueError("Component adapter input has no train rows.")
    torch.manual_seed(seed)
    x = torch.tensor(frame[columns].to_numpy(dtype="float32"), dtype=torch.float32)
    y = torch.tensor(frame[label_col].astype(int).to_numpy(), dtype=torch.long)
    env = torch.tensor(frame[env_col].astype(int).to_numpy(), dtype=torch.long)
    train_indices = torch.nonzero(torch.tensor(train_mask.to_numpy(), dtype=torch.bool), as_tuple=False).flatten()
    train_x = x[train_indices]
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    x_norm = (x - mean) / std
    train_x_norm = x_norm[train_indices]
    train_y = y[train_indices]
    train_env = env[train_indices]
    class_count = int(y.max().item()) + 1
    priors = adapter_priors_from_clues(columns, clue_rows).clamp(0.02, 0.98)
    gate_logits = torch.nn.Parameter(torch.logit(priors))
    classifier = torch.nn.Linear(len(columns), class_count)
    env_class_count = int(env.max().item()) + 1
    env_classifier = torch.nn.Linear(len(columns), env_class_count)
    optimizer = torch.optim.Adam([gate_logits, *classifier.parameters()], lr=lr)
    env_optimizer = torch.optim.Adam(env_classifier.parameters(), lr=lr)
    train_weights = _group_balanced_weights(frame.loc[train_mask]).to(dtype=torch.float32)

    for _ in range(max(0, int(epochs))):
        gate = torch.sigmoid(gate_logits)
        adapted_train = train_x_norm * gate
        env_logits = env_classifier(adapted_train.detach())
        env_fit_loss = torch.nn.functional.cross_entropy(env_logits, train_env)
        env_optimizer.zero_grad()
        env_fit_loss.backward()
        env_optimizer.step()

        gate = torch.sigmoid(gate_logits)
        adapted_train = train_x_norm * gate
        logits = classifier(adapted_train)
        per_row_loss = torch.nn.functional.cross_entropy(logits, train_y, reduction="none")
        label_loss = (per_row_loss * train_weights).mean()
        env_penalty = _env_covariance_penalty(adapted_train, train_env)
        prior_loss = torch.nn.functional.mse_loss(gate, priors)
        for parameter in env_classifier.parameters():
            parameter.requires_grad_(False)
        env_logits = env_classifier(adapted_train)
        env_loss = torch.nn.functional.cross_entropy(env_logits, train_env)
        loss = (
            label_loss
            + float(env_penalty_weight) * env_penalty
            + float(clue_prior_weight) * prior_loss
            - float(env_adversary_weight) * env_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for parameter in env_classifier.parameters():
            parameter.requires_grad_(True)

    with torch.no_grad():
        gate = torch.sigmoid(gate_logits)
        adapted = x_norm * gate
        train_logits = classifier(adapted[train_indices])
        train_accuracy = float((train_logits.argmax(dim=1) == train_y).float().mean().item())
        train_env_logits = env_classifier(adapted[train_indices])
        train_env_accuracy = float((train_env_logits.argmax(dim=1) == train_env).float().mean().item())
        env_penalty = float(_env_covariance_penalty(adapted[train_indices], train_env).item())

    adapted_columns = [f"{output_prefix}_{index:04d}" for index in range(len(columns))]
    output = frame.loc[:, _metadata_columns(frame, columns)].copy()
    adapted_frame = pd.DataFrame(adapted.detach().cpu().numpy(), columns=adapted_columns)
    output = pd.concat([output.reset_index(drop=True), adapted_frame], axis=1)
    report = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "input_feature_count": len(columns),
        "output_feature_count": len(adapted_columns),
        "epochs": int(epochs),
        "lr": float(lr),
        "env_penalty_weight": float(env_penalty_weight),
        "env_adversary_weight": float(env_adversary_weight),
        "clue_prior_weight": float(clue_prior_weight),
        "train_accuracy": train_accuracy,
        "train_env_accuracy": train_env_accuracy,
        "train_env_abs_corr": env_penalty,
        "feature_columns": columns,
        "adapted_feature_columns": adapted_columns,
        "feature_weights": {column: float(value) for column, value in zip(columns, gate.tolist(), strict=True)},
        "feature_priors": {column: float(value) for column, value in zip(columns, priors.tolist(), strict=True)},
    }
    return ComponentAdapterResult(feature_frame=output, report=report)


def write_component_manifest(
    *,
    output_csv: Path,
    feature_columns: Sequence[str],
    feature_components: Mapping[str, Sequence[str]],
    source_path: Path,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    manifest_path = output_csv.with_suffix(output_csv.suffix + ".manifest.json")
    payload = {
        "schema_version": COMPONENT_REPRESENTATION_SCHEMA_VERSION,
        "feature_extractor": "component_representation_compiler",
        "feature_source": str(source_path),
        "split_definition": "copied from source feature table",
        "base_metrics": {},
        "resolved_settings": dict(extra or {}),
        "feature_columns": [str(column) for column in feature_columns],
        "feature_components": {
            str(key): [str(item) for item in value]
            for key, value in feature_components.items()
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any, NamedTuple

import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.config import load_config
from causality_experiments.counterfactual_clue_tests import execute_clue_test
from causality_experiments.data import load_dataset
from causality_experiments.llm_clue_planner import ClueTestSpec
from causality_experiments.run import run_experiment
from causality_experiments.sklearn_compat import LogisticRegression, StandardScaler
from scripts.run_waterbirds_clue_fusion_sweep import (
    build_bridge_score_rows,
    build_downstream_candidate,
    build_pairwise_bridge_score_rows,
    build_policy_score_rows,
    build_source_score_rows,
    with_dataset_path,
    with_runtime_overrides,
)
from causality_experiments.discovery import build_feature_clue_rows
from causality_experiments.clues import write_csv_rows


def _float_values(values: list[str] | None, default: list[float]) -> list[float]:
    if not values:
        return default
    return [float(value) for value in values]


def _int_values(values: list[str] | None, default: list[int]) -> list[int]:
    if not values:
        return default
    return [int(value) for value in values]


def _weight_label(weight: float) -> str:
    return f"w{weight:g}".replace(".", "p").replace("-", "m")


def _metric_payload(run_dir: Path) -> dict[str, float]:
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return dict(payload["metrics"])


def _stable_random_score(*, control_index: int, feature_index: str, feature_name: str) -> float:
    payload = f"{control_index}:{feature_index}:{feature_name}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return int(digest, 16) / float(16**16 - 1)


def build_random_score_rows(clue_rows: list[dict[str, Any]], *, control_index: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in clue_rows:
        feature_index = str(row.get("feature_index", ""))
        feature_name = str(row.get("feature_name", ""))
        score = _stable_random_score(
            control_index=control_index,
            feature_index=feature_index,
            feature_name=feature_name,
        )
        rows.append(
            {
                "dataset": str(row.get("dataset", "")),
                "feature_index": feature_index,
                "feature_name": feature_name,
                "support_score": f"{score:.6f}",
                "rank_score": f"{score:.6f}",
                "score": f"{score:.6f}",
                "score_source": f"random_score_{control_index}",
            }
        )
    return rows


def _safe_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _normalise_score_map(rows: list[dict[str, str]]) -> dict[str, float]:
    values = {row["feature_name"]: _safe_float(row, "score") for row in rows}
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high - low < 1e-12:
        return {key: 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


RISK_FEATURE_COLUMNS = (
    "label_corr",
    "env_corr",
    "corr_margin",
    "abs_corr_margin",
    "uncertainty",
    "top_group_entropy",
    "label_env_disentanglement",
)


class ArtifactRiskHead(NamedTuple):
    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    train_trace_count: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dataset_name(run_dir: Path) -> str:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = str(manifest.get("config", ""))
        if config:
            return Path(config).stem
        dataset = str(manifest.get("dataset", ""))
        if dataset:
            return dataset
    return run_dir.name


def _matches_any_dataset_name(dataset: str, patterns: list[str] | None) -> bool:
    if not patterns:
        return False
    lowered = dataset.lower()
    return any(pattern.strip().lower() in lowered for pattern in patterns if pattern.strip())


def _risk_feature_vector(row: dict[str, Any]) -> list[float]:
    label_corr = _safe_float(row, "label_corr")
    env_corr = _safe_float(row, "env_corr")
    corr_margin = _safe_float(row, "corr_margin")
    values = [_safe_float(row, column) for column in RISK_FEATURE_COLUMNS]
    values.extend(
        [
            label_corr - env_corr,
            max(0.0, env_corr - label_corr),
            1.0,
        ]
    )
    return values


def _risk_target(row: dict[str, Any]) -> float:
    label_corr = _safe_float(row, "label_corr")
    env_corr = _safe_float(row, "env_corr")
    shortcut_excess = max(0.0, env_corr - label_corr)
    geometry_risk = shortcut_excess / max(abs(label_corr) + abs(env_corr), 1e-6)
    failed_control = 0.0 if bool(row.get("passed_control", False)) else 1.0
    wrong_hypothesis = 0.0 if bool(row.get("hypothesis_correct", False)) else 1.0
    weak_delta = 1.0 if _safe_float(row, "score_delta") <= 0.0 else 0.0
    low_test_value = 1.0 if _safe_float(row, "test_value") <= 0.0 else 0.0
    target = 0.40 * geometry_risk + 0.20 * failed_control + 0.15 * wrong_hypothesis + 0.15 * weak_delta + 0.10 * low_test_value
    return min(max(target, 0.0), 1.0)


def fit_artifact_risk_head(
    input_dir: Path,
    *,
    alpha: float = 10.0,
    exclude_datasets: list[str] | None = None,
) -> ArtifactRiskHead:
    run_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not run_dirs:
        raise ValueError(f"No run directories found under {input_dir}.")
    traces: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        dataset = _dataset_name(run_dir)
        if _matches_any_dataset_name(dataset, exclude_datasets):
            continue
        traces.extend(_read_jsonl(run_dir / "training_traces.jsonl"))
    if not traces:
        raise ValueError(f"No artifact-risk training traces found under {input_dir}.")
    x = np.asarray([_risk_feature_vector(row) for row in traces], dtype=np.float64)
    y = np.asarray([_risk_target(row) for row in traces], dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    mean[-1] = 0.0
    scale[-1] = 1.0
    xz = (x - mean) / scale
    penalty = float(alpha) * np.eye(xz.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    weights = np.linalg.pinv(xz.T @ xz + penalty) @ xz.T @ y
    return ArtifactRiskHead(weights=weights, mean=mean, scale=scale, train_trace_count=len(traces))


def _predict_artifact_risk(row: dict[str, Any], risk_head: ArtifactRiskHead) -> float:
    x = np.asarray(_risk_feature_vector(row), dtype=np.float64)
    raw = float(((x - risk_head.mean) / risk_head.scale) @ risk_head.weights)
    return min(max(raw, 0.0), 1.0)


def build_artifact_risk_score_rows(
    *,
    clue_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, str]],
    risk_head: ArtifactRiskHead,
    top_k: int,
    risk_weight: float = 0.25,
    boundary_fraction: float | None = None,
) -> list[dict[str, str]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive for artifact-risk scoring.")
    clues_by_feature = {str(row.get("feature_name", "")): row for row in clue_rows}
    base_norm = _normalise_score_map(candidate_rows)
    names = [row["feature_name"] for row in candidate_rows]
    ranked_names = sorted(names, key=lambda name: base_norm.get(name, 0.0), reverse=True)
    if boundary_fraction is None:
        boundary_names = set(names)
        source = "artifact_risk"
    else:
        window = max(1, int(round(float(boundary_fraction) * top_k)))
        low_rank = max(0, top_k - window)
        high_rank = min(len(ranked_names), top_k + window)
        boundary_names = set(ranked_names[low_rank:high_rank])
        source = "artifact_risk_boundary"
    rows: list[dict[str, str]] = []
    for row in candidate_rows:
        name = row["feature_name"]
        clue = clues_by_feature.get(name, {})
        risk = _predict_artifact_risk(clue, risk_head)
        penalty = float(risk_weight) * risk if name in boundary_names else 0.0
        score = base_norm.get(name, 0.0) - penalty
        updated = dict(row)
        updated["artifact_risk"] = f"{risk:.6f}"
        updated["support_score"] = f"{score:.6f}"
        updated["rank_score"] = f"{score:.6f}"
        updated["score"] = f"{score:.6f}"
        updated["score_source"] = source
        rows.append(updated)
    return rows


def _is_env_dominant(clue: dict[str, Any]) -> bool:
    return _safe_float(clue, "env_corr") >= _safe_float(clue, "label_corr")


def build_constrained_support_score_rows(
    *,
    clue_rows: list[dict[str, Any]],
    stats_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    top_k: int,
    stats_core_fraction: float = 0.6,
    env_dominant_cap: int | None = None,
) -> list[dict[str, str]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive for constrained support selection.")
    stats_by_feature = {row["feature_name"]: row for row in stats_rows}
    candidate_by_feature = {row["feature_name"]: row for row in candidate_rows}
    clues_by_feature = {str(row.get("feature_name", "")): row for row in clue_rows}
    stats_norm = _normalise_score_map(stats_rows)
    candidate_norm = _normalise_score_map(candidate_rows)
    names = [row["feature_name"] for row in candidate_rows]
    ranked_stats = sorted(names, key=lambda name: stats_norm.get(name, 0.0), reverse=True)
    ranked_candidates = sorted(
        names,
        key=lambda name: (
            candidate_norm.get(name, 0.0) + 0.15 * stats_norm.get(name, 0.0) - (0.20 if _is_env_dominant(clues_by_feature.get(name, {})) else 0.0),
            stats_norm.get(name, 0.0),
        ),
        reverse=True,
    )
    env_cap = max(0, int(env_dominant_cap if env_dominant_cap is not None else round(0.02 * top_k)))
    stats_core_count = min(top_k, max(0, int(round(float(stats_core_fraction) * top_k))))
    selected: list[str] = []
    selected_set: set[str] = set()
    env_count = 0

    def try_add(name: str, *, enforce_env_cap: bool) -> bool:
        nonlocal env_count
        if name in selected_set:
            return False
        clue = clues_by_feature.get(name, {})
        env_dominant = _is_env_dominant(clue)
        if enforce_env_cap and env_dominant and env_count >= env_cap:
            return False
        selected.append(name)
        selected_set.add(name)
        if env_dominant:
            env_count += 1
        return True

    for name in ranked_stats:
        if len(selected) >= stats_core_count:
            break
        try_add(name, enforce_env_cap=True)

    for name in ranked_candidates:
        if len(selected) >= top_k:
            break
        try_add(name, enforce_env_cap=True)

    for name in ranked_stats:
        if len(selected) >= top_k:
            break
        try_add(name, enforce_env_cap=False)

    selected_rank = {name: index for index, name in enumerate(selected[:top_k])}
    rows: list[dict[str, str]] = []
    denominator = max(len(selected_rank), 1)
    for name in names:
        source = candidate_by_feature.get(name, stats_by_feature.get(name, {}))
        if name in selected_rank:
            score = 2.0 + (denominator - selected_rank[name]) / denominator
        else:
            score = 0.1 * (candidate_norm.get(name, 0.0) + stats_norm.get(name, 0.0))
        updated = dict(source)
        updated["support_score"] = f"{score:.6f}"
        updated["rank_score"] = f"{score:.6f}"
        updated["score"] = f"{score:.6f}"
        updated["score_source"] = "constrained_support"
        rows.append(updated)
    return rows


def _constrained_variant_params(variant_key: str, top_k: int) -> tuple[float, int]:
    if variant_key == "constrained_support_strict":
        return 0.75, max(1, round(0.01 * top_k))
    if variant_key == "constrained_support_loose":
        return 0.45, max(1, round(0.04 * top_k))
    if variant_key == "constrained_support_bridge":
        return 0.30, max(1, round(0.02 * top_k))
    return 0.60, max(1, round(0.02 * top_k))


def _is_constrained_support_variant(variant_key: str) -> bool:
    return variant_key in {
        "constrained_support",
        "constrained_support_strict",
        "constrained_support_loose",
        "constrained_support_bridge",
    }


def _is_artifact_risk_variant(variant_key: str) -> bool:
    return variant_key in {"artifact_risk", "artifact_risk_boundary"}


def build_active_boundary_score_rows(
    *,
    bundle: Any,
    clue_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, str]],
    top_k: int,
    boundary_fraction: float = 0.15,
    evidence_weight: float = 0.25,
    split_name: str = "train",
) -> list[dict[str, str]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive for active-boundary scoring.")
    base_norm = _normalise_score_map(candidate_rows)
    names = [row["feature_name"] for row in candidate_rows]
    ranked_names = sorted(names, key=lambda name: base_norm.get(name, 0.0), reverse=True)
    window = max(1, int(round(float(boundary_fraction) * top_k)))
    low_rank = max(0, top_k - window)
    high_rank = min(len(ranked_names), top_k + window)
    boundary_names = set(ranked_names[low_rank:high_rank])
    clues_by_feature = {str(row.get("feature_name", "")): row for row in clue_rows}
    raw_evidence: dict[str, float] = {}
    result_by_feature: dict[str, dict[str, Any]] = {}
    for name in sorted(boundary_names):
        clue = clues_by_feature.get(name, {})
        spec = ClueTestSpec(
            candidate_id=str(clue.get("candidate_id", name)),
            feature_name=name,
            action="conditional_signal_check",
            expected_direction="positive",
            control="next_feature",
            cost=0.25,
            reason_code="active_boundary_conditional_signal",
            evidence_ids=(),
        )
        result = execute_clue_test(bundle, spec, packet=clue, model=None, split_name=split_name)
        label_delta = _safe_float(result, "test_effect_label_delta")
        env_delta = abs(_safe_float(result, "test_effect_env_delta"))
        random_delta = abs(_safe_float(result, "test_random_control_delta"))
        selectivity = _safe_float(result, "test_effect_selectivity")
        evidence = label_delta + selectivity - env_delta - random_delta
        raw_evidence[name] = evidence
        result_by_feature[name] = result
    if raw_evidence:
        low = min(raw_evidence.values())
        high = max(raw_evidence.values())
        if high - low < 1e-12:
            evidence_norm = {name: 0.5 for name in raw_evidence}
        else:
            evidence_norm = {name: (value - low) / (high - low) for name, value in raw_evidence.items()}
    else:
        evidence_norm = {}
    rows: list[dict[str, str]] = []
    for row in candidate_rows:
        name = row["feature_name"]
        adjustment = float(evidence_weight) * (evidence_norm.get(name, 0.5) - 0.5) if name in boundary_names else 0.0
        score = base_norm.get(name, 0.0) + adjustment
        result = result_by_feature.get(name, {})
        updated = dict(row)
        updated["active_boundary_evidence"] = f"{raw_evidence.get(name, 0.0):.6f}"
        updated["active_boundary_label_delta"] = f"{_safe_float(result, 'test_effect_label_delta'):.6f}"
        updated["active_boundary_env_delta"] = f"{_safe_float(result, 'test_effect_env_delta'):.6f}"
        updated["active_boundary_random_delta"] = f"{_safe_float(result, 'test_random_control_delta'):.6f}"
        updated["support_score"] = f"{score:.6f}"
        updated["rank_score"] = f"{score:.6f}"
        updated["score"] = f"{score:.6f}"
        updated["score_source"] = "active_boundary"
        rows.append(updated)
    return rows


def _feature_indices_by_name(bundle: Any, rows: list[dict[str, Any]]) -> dict[str, int]:
    feature_columns = list((getattr(bundle, "metadata", None) or {}).get("feature_columns", []) or [])
    by_name = {str(name): index for index, name in enumerate(feature_columns)}
    for row in rows:
        name = str(row.get("feature_name", ""))
        if not name or name in by_name:
            continue
        value = str(row.get("feature_index", "")).strip()
        try:
            index = int(value)
        except ValueError:
            continue
        if 0 <= index < int(getattr(bundle, "input_dim", 0)):
            by_name[name] = index
    return by_name


def _balanced_probe_split(groups: np.ndarray, *, seed: int, eval_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    eval_parts: list[np.ndarray] = []
    for group in sorted(int(value) for value in np.unique(groups)):
        indices = np.flatnonzero(groups == group)
        rng.shuffle(indices)
        eval_count = min(len(indices) - 1, max(1, int(round(float(eval_fraction) * len(indices)))))
        if eval_count <= 0:
            train_parts.append(indices)
            continue
        eval_parts.append(indices[:eval_count])
        train_parts.append(indices[eval_count:])
    if not train_parts or not eval_parts:
        indices = np.arange(len(groups))
        rng.shuffle(indices)
        split_at = max(1, min(len(indices) - 1, int(round((1.0 - float(eval_fraction)) * len(indices)))))
        return indices[:split_at], indices[split_at:]
    return np.concatenate(train_parts), np.concatenate(eval_parts)


def _binary_log_loss_from_scores(scores: np.ndarray, y: np.ndarray, positive_label: int) -> float:
    target = (y == int(positive_label)).astype(np.float64)
    return float(np.mean(np.maximum(scores, 0.0) - scores * target + np.log1p(np.exp(-np.abs(scores)))))


def _worst_group_accuracy_from_predictions(pred: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    scores: list[float] = []
    for group in sorted(int(value) for value in np.unique(groups)):
        mask = groups == group
        if np.any(mask):
            scores.append(float(np.mean(pred[mask] == y[mask])))
    return min(scores) if scores else 0.0


def build_active_boundary_model_effect_score_rows(
    *,
    bundle: Any,
    clue_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, str]],
    top_k: int,
    boundary_fraction: float = 0.15,
    evidence_weight: float = 0.35,
    probe_seed: int = 17,
    eval_fraction: float = 0.35,
) -> list[dict[str, str]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive for active-boundary model-effect scoring.")
    base_norm = _normalise_score_map(candidate_rows)
    names = [row["feature_name"] for row in candidate_rows]
    ranked_names = sorted(names, key=lambda name: base_norm.get(name, 0.0), reverse=True)
    window = max(1, int(round(float(boundary_fraction) * top_k)))
    low_rank = max(0, top_k - window)
    high_rank = min(len(ranked_names), top_k + window)
    boundary_names = set(ranked_names[low_rank:high_rank])
    support_names = set(ranked_names[: min(top_k, len(ranked_names))])
    probe_names = list(dict.fromkeys([name for name in ranked_names if name in support_names or name in boundary_names]))
    feature_indices = _feature_indices_by_name(bundle, [*clue_rows, *candidate_rows])
    resolved_probe_names = [name for name in probe_names if name in feature_indices]
    probe_indices = [feature_indices[name] for name in resolved_probe_names]
    if not probe_indices:
        raise ValueError("active-boundary model-effect scoring could not resolve any feature indices.")

    train_split = bundle.split("train")
    x = train_split["x"].detach().cpu().numpy().astype(np.float64, copy=False)
    y = train_split["y"].detach().cpu().numpy().astype(np.int64, copy=False)
    groups = train_split["group"].detach().cpu().numpy().astype(np.int64, copy=False)
    train_idx, eval_idx = _balanced_probe_split(groups, seed=probe_seed, eval_fraction=eval_fraction)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx][:, probe_indices])
    x_eval = scaler.transform(x[eval_idx][:, probe_indices])
    y_train = y[train_idx]
    y_eval = y[eval_idx]
    group_eval = groups[eval_idx]
    class_counts = {int(label): int(np.sum(y_train == label)) for label in np.unique(y_train)}
    class_weight = {
        label: len(y_train) / max(len(class_counts) * count, 1)
        for label, count in class_counts.items()
    }
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=0.3,
        max_iter=1000,
        random_state=probe_seed,
        class_weight=class_weight,
    )
    model.fit(x_train, y_train)
    classes = [int(value) for value in getattr(model, "classes_", np.unique(y_train))]
    if len(classes) != 2:
        raise ValueError("active-boundary model-effect scoring currently supports binary classification.")
    positive_label = classes[1]
    base_pred = model.predict(x_eval)
    base_wga = _worst_group_accuracy_from_predictions(base_pred, y_eval, group_eval)
    base_scores = np.asarray(model.decision_function(x_eval), dtype=np.float64)
    base_loss = _binary_log_loss_from_scores(base_scores, y_eval, positive_label)
    coefficients = np.asarray(getattr(model, "coef_", np.zeros((1, len(probe_indices)))), dtype=np.float64)
    coefficient_abs = np.abs(coefficients).mean(axis=0)
    coefficient_max = float(np.max(coefficient_abs)) if coefficient_abs.size else 0.0
    local_index_by_name = {name: index for index, name in enumerate(resolved_probe_names)}

    raw_evidence: dict[str, float] = {}
    details_by_feature: dict[str, tuple[float, float, float]] = {}
    for name in sorted(boundary_names):
        local_index = local_index_by_name.get(name)
        if local_index is None or local_index >= x_eval.shape[1]:
            raw_evidence[name] = 0.0
            details_by_feature[name] = (0.0, 0.0, 0.0)
            continue
        ablated = np.array(x_eval, copy=True)
        ablated[:, local_index] = 0.0
        ablated_pred = model.predict(ablated)
        ablated_wga = _worst_group_accuracy_from_predictions(ablated_pred, y_eval, group_eval)
        ablated_scores = np.asarray(model.decision_function(ablated), dtype=np.float64)
        ablated_loss = _binary_log_loss_from_scores(ablated_scores, y_eval, positive_label)
        wga_effect = base_wga - ablated_wga
        loss_effect = ablated_loss - base_loss
        coef_effect = float(coefficient_abs[local_index] / coefficient_max) if coefficient_max > 1e-12 else 0.0
        evidence = wga_effect + 0.25 * loss_effect + 0.05 * coef_effect
        raw_evidence[name] = evidence
        details_by_feature[name] = (wga_effect, loss_effect, coef_effect)
    if raw_evidence:
        low = min(raw_evidence.values())
        high = max(raw_evidence.values())
        if high - low < 1e-12:
            evidence_norm = {name: 0.5 for name in raw_evidence}
        else:
            evidence_norm = {name: (value - low) / (high - low) for name, value in raw_evidence.items()}
    else:
        evidence_norm = {}

    rows: list[dict[str, str]] = []
    for row in candidate_rows:
        name = row["feature_name"]
        adjustment = float(evidence_weight) * (evidence_norm.get(name, 0.5) - 0.5) if name in boundary_names else 0.0
        score = base_norm.get(name, 0.0) + adjustment
        wga_effect, loss_effect, coef_effect = details_by_feature.get(name, (0.0, 0.0, 0.0))
        updated = dict(row)
        updated["active_boundary_model_effect"] = f"{raw_evidence.get(name, 0.0):.6f}"
        updated["active_boundary_wga_effect"] = f"{wga_effect:.6f}"
        updated["active_boundary_loss_effect"] = f"{loss_effect:.6f}"
        updated["active_boundary_coef_effect"] = f"{coef_effect:.6f}"
        updated["support_score"] = f"{score:.6f}"
        updated["rank_score"] = f"{score:.6f}"
        updated["score"] = f"{score:.6f}"
        updated["score_source"] = "active_boundary_model_effect"
        rows.append(updated)
    return rows


def _is_active_boundary_variant(variant_key: str) -> bool:
    return variant_key in {"active_boundary", "active_boundary_model_effect"}


def build_support_variant_score_rows(
    *,
    clue_rows: list[dict[str, Any]],
    stats_rows: list[dict[str, str]],
    bridge_rows: list[dict[str, str]],
    variant: str,
) -> list[dict[str, str]]:
    stats_by_feature = {row["feature_name"]: _safe_float(row, "score") for row in stats_rows}
    clues_by_feature = {str(row.get("feature_name", "")): row for row in clue_rows}
    variant_key = variant.strip().lower()
    rows: list[dict[str, str]] = []
    for row in bridge_rows:
        feature_name = row["feature_name"]
        clue = clues_by_feature.get(feature_name, {})
        bridge_score = _safe_float(row, "score")
        stats_score = stats_by_feature.get(feature_name, 0.0)
        label_corr = _safe_float(clue, "label_corr")
        env_corr = _safe_float(clue, "env_corr")
        corr_margin = _safe_float(clue, "corr_margin")
        shortcut_excess = max(0.0, env_corr - label_corr)
        if variant_key == "env_filter":
            score = bridge_score if label_corr > env_corr else 0.0
        elif variant_key == "margin_gate":
            score = bridge_score * max(0.0, min(1.0, 0.5 + 0.5 * corr_margin))
        elif variant_key == "stats_fill":
            score = bridge_score if label_corr > env_corr else 0.5 * stats_score
        elif variant_key == "soft_env_penalty":
            score = bridge_score * max(0.25, 1.0 - 0.5 * shortcut_excess)
        elif variant_key == "stats_anchor":
            score = 0.75 * bridge_score + 0.25 * stats_score if label_corr > env_corr else 0.5 * bridge_score + 0.5 * stats_score
        elif variant_key == "score_sqrt":
            score = math.sqrt(max(0.0, bridge_score))
        elif variant_key == "score_square":
            score = max(0.0, bridge_score) ** 2
        else:
            raise ValueError(
                "support variant must be one of: env_filter, margin_gate, stats_fill, "
                "soft_env_penalty, stats_anchor, score_sqrt, score_square."
            )
        updated = dict(row)
        updated["support_score"] = f"{score:.6f}"
        updated["rank_score"] = f"{score:.6f}"
        updated["score"] = f"{score:.6f}"
        updated["score_source"] = f"bridge_fused_{variant_key}"
        rows.append(updated)
    return rows


def _row(
    *,
    row_type: str,
    label: str,
    seed: int,
    top_k: int,
    weight: float | None,
    run_dir: Path,
    metrics: dict[str, float],
    baseline_metrics: dict[str, float] | None,
    stats_metrics: dict[str, float] | None,
) -> dict[str, str]:
    test_wga = float(metrics.get("test/worst_group_accuracy", 0.0))
    baseline_wga = None if baseline_metrics is None else float(baseline_metrics.get("test/worst_group_accuracy", 0.0))
    stats_wga = None if stats_metrics is None else float(stats_metrics.get("test/worst_group_accuracy", 0.0))
    return {
        "row_type": row_type,
        "label": label,
        "seed": str(seed),
        "top_k": str(top_k),
        "bridge_fused_weight": "" if weight is None else str(weight),
        "run": run_dir.name,
        "val_wga": str(metrics.get("val/worst_group_accuracy", "")),
        "test_wga": str(test_wga),
        "val_acc": str(metrics.get("val/accuracy", "")),
        "test_acc": str(metrics.get("test/accuracy", "")),
        "baseline_test_wga": "" if baseline_wga is None else str(baseline_wga),
        "delta_to_baseline": "" if baseline_wga is None else str(test_wga - baseline_wga),
        "stats_test_wga": "" if stats_wga is None else str(stats_wga),
        "delta_to_stats": "" if stats_wga is None else str(test_wga - stats_wga),
    }


def _summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, str]]] = {}
    random_groups: dict[str, list[dict[str, str]]] = {}
    best_random_by_seed_top_k: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["row_type"] == "candidate":
            groups.setdefault(row["label"], []).append(row)
        elif row["row_type"] == "random_control":
            random_groups.setdefault(row["label"], []).append(row)
            key = (row["seed"], row["top_k"])
            best_random_by_seed_top_k[key] = max(
                best_random_by_seed_top_k.get(key, float("-inf")),
                float(row["test_wga"]),
            )
    candidates: list[dict[str, Any]] = []
    for label, items in sorted(groups.items()):
        wgas = [float(item["test_wga"]) for item in items]
        baseline_deltas = [float(item["delta_to_baseline"]) for item in items if item["delta_to_baseline"]]
        stats_deltas = [float(item["delta_to_stats"]) for item in items if item["delta_to_stats"]]
        random_deltas = [
            float(item["test_wga"]) - best_random_by_seed_top_k[(item["seed"], item["top_k"])]
            for item in items
            if (item["seed"], item["top_k"]) in best_random_by_seed_top_k
        ]
        candidates.append(
            {
                "label": label,
                "count": len(items),
                "mean_test_wga": statistics.mean(wgas),
                "min_test_wga": min(wgas),
                "max_test_wga": max(wgas),
                "mean_delta_to_baseline": statistics.mean(baseline_deltas) if baseline_deltas else 0.0,
                "min_delta_to_baseline": min(baseline_deltas) if baseline_deltas else 0.0,
                "mean_delta_to_stats": statistics.mean(stats_deltas) if stats_deltas else 0.0,
                "min_delta_to_stats": min(stats_deltas) if stats_deltas else 0.0,
                "mean_delta_to_best_random": statistics.mean(random_deltas) if random_deltas else 0.0,
                "min_delta_to_best_random": min(random_deltas) if random_deltas else 0.0,
                "non_negative_baseline_seeds": sum(delta >= 0.0 for delta in baseline_deltas),
                "non_negative_stats_seeds": sum(delta >= 0.0 for delta in stats_deltas),
                "non_negative_best_random_seeds": sum(delta >= 0.0 for delta in random_deltas),
            }
        )
    candidates.sort(key=lambda item: (item["mean_delta_to_baseline"], item["mean_delta_to_stats"]), reverse=True)
    random_controls: list[dict[str, Any]] = []
    for label, items in sorted(random_groups.items()):
        wgas = [float(item["test_wga"]) for item in items]
        baseline_deltas = [float(item["delta_to_baseline"]) for item in items if item["delta_to_baseline"]]
        stats_deltas = [float(item["delta_to_stats"]) for item in items if item["delta_to_stats"]]
        random_controls.append(
            {
                "label": label,
                "count": len(items),
                "mean_test_wga": statistics.mean(wgas),
                "min_test_wga": min(wgas),
                "max_test_wga": max(wgas),
                "mean_delta_to_baseline": statistics.mean(baseline_deltas) if baseline_deltas else 0.0,
                "min_delta_to_baseline": min(baseline_deltas) if baseline_deltas else 0.0,
                "mean_delta_to_stats": statistics.mean(stats_deltas) if stats_deltas else 0.0,
                "min_delta_to_stats": min(stats_deltas) if stats_deltas else 0.0,
            }
        )
    random_controls.sort(key=lambda item: (item["mean_delta_to_baseline"], item["mean_delta_to_stats"]), reverse=True)
    return {"candidates": candidates, "random_controls": random_controls}


def run_bridge_fused_sweep(
    *,
    baseline_config_path: Path,
    candidate_config_path: Path,
    dataset_path: str,
    bridge_input_dir: Path,
    out_dir: Path,
    output_csv: Path,
    output_json: Path,
    seeds: list[int],
    top_k_values: list[int],
    bridge_fused_weights: list[float],
    support_variants: list[str],
    bridge_score_source: str,
    bridge_alpha: float,
    bridge_exclude_datasets: list[str],
    policy_input_dir: Path,
    policy_alpha: float,
    policy_exclude_datasets: list[str],
    card_top_k: int,
    random_control_count: int,
    num_retrains: int,
    training_device: str | None,
    output_root: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    base_for_scores = with_runtime_overrides(
        with_dataset_path(load_config(candidate_config_path), dataset_path),
        official_dfr_num_retrains=num_retrains,
        training_device=training_device,
    )
    bundle = load_dataset(base_for_scores)
    clue_rows = build_feature_clue_rows(bundle, split_name="train")
    stats_path = out_dir / "scores_stats.csv"
    stats_rows = build_source_score_rows(clue_rows, "stats")
    write_csv_rows(stats_path, stats_rows)
    random_paths: dict[int, Path] = {}
    for control_index in range(max(0, int(random_control_count))):
        score_path = out_dir / f"scores_random_control_{control_index}.csv"
        write_csv_rows(score_path, build_random_score_rows(clue_rows, control_index=control_index))
        random_paths[control_index] = score_path
    bridge_paths: dict[float | tuple[float, str], Path] = {}
    source = bridge_score_source.strip().lower()
    if source not in {"bridge_fused", "bridge_gated", "pairwise_bridge_fused", "policy_fused", "policy_safe"}:
        raise ValueError(
            "bridge_score_source must be 'bridge_fused', 'bridge_gated', 'pairwise_bridge_fused', 'policy_fused', or 'policy_safe'."
        )
    risk_head: ArtifactRiskHead | None = None
    for weight in bridge_fused_weights:
        score_path = out_dir / f"scores_{source}_{_weight_label(weight)}.csv"
        if source in {"bridge_fused", "bridge_gated"}:
            bridge_rows = build_bridge_score_rows(
                bundle,
                bridge_input_dir=bridge_input_dir,
                alpha=bridge_alpha,
                exclude_datasets=bridge_exclude_datasets,
                split_name="train",
                card_top_k=card_top_k,
                blend_with_stats_weight=weight,
                blend_mode="gated" if source == "bridge_gated" else "linear",
            )
        elif source == "pairwise_bridge_fused":
            bridge_rows = build_pairwise_bridge_score_rows(
                bundle,
                bridge_input_dir=bridge_input_dir,
                alpha=bridge_alpha,
                exclude_datasets=bridge_exclude_datasets,
                split_name="train",
                card_top_k=card_top_k,
                blend_with_stats_weight=weight,
            )
        else:
            bridge_rows = build_policy_score_rows(
                bundle,
                policy_input_dir=policy_input_dir,
                alpha=policy_alpha,
                exclude_datasets=policy_exclude_datasets,
                split_name="train",
                card_top_k=card_top_k,
                blend_with_stats_weight=weight,
                blend_mode="safe_residual" if source == "policy_safe" else "minmax",
            )
        write_csv_rows(score_path, bridge_rows)
        bridge_paths[weight] = score_path
        if source == "bridge_fused":
            for variant in support_variants:
                variant_key = variant.strip().lower()
                if variant_key not in {
                    "env_filter",
                    "margin_gate",
                    "stats_fill",
                    "soft_env_penalty",
                    "stats_anchor",
                    "score_sqrt",
                    "score_square",
                    "constrained_support",
                    "constrained_support_strict",
                    "constrained_support_loose",
                    "constrained_support_bridge",
                    "artifact_risk",
                    "artifact_risk_boundary",
                    "active_boundary",
                    "active_boundary_model_effect",
                }:
                    raise ValueError(
                        "support variants must be one of: env_filter, margin_gate, stats_fill, "
                        "soft_env_penalty, stats_anchor, score_sqrt, score_square, constrained_support, "
                        "constrained_support_strict, constrained_support_loose, constrained_support_bridge, "
                        "artifact_risk, artifact_risk_boundary, active_boundary, active_boundary_model_effect."
                    )
                if _is_active_boundary_variant(variant_key):
                    for top_k in top_k_values:
                        variant_path = out_dir / f"scores_{source}_{_weight_label(weight)}_{variant_key}_top{top_k}.csv"
                        if variant_key == "active_boundary_model_effect":
                            variant_rows = build_active_boundary_model_effect_score_rows(
                                bundle=bundle,
                                clue_rows=clue_rows,
                                candidate_rows=bridge_rows,
                                top_k=top_k,
                                boundary_fraction=0.15,
                                evidence_weight=0.35,
                            )
                        else:
                            variant_rows = build_active_boundary_score_rows(
                                bundle=bundle,
                                clue_rows=clue_rows,
                                candidate_rows=bridge_rows,
                                top_k=top_k,
                                boundary_fraction=0.15,
                                evidence_weight=0.25,
                                split_name="train",
                            )
                        write_csv_rows(
                            variant_path,
                            variant_rows,
                        )
                        bridge_paths[(weight, variant_key, int(top_k))] = variant_path
                    continue
                if _is_artifact_risk_variant(variant_key):
                    if risk_head is None:
                        risk_head = fit_artifact_risk_head(
                            bridge_input_dir,
                            alpha=bridge_alpha,
                            exclude_datasets=bridge_exclude_datasets,
                        )
                    for top_k in top_k_values:
                        boundary_fraction = 0.15 if variant_key == "artifact_risk_boundary" else None
                        variant_path = out_dir / f"scores_{source}_{_weight_label(weight)}_{variant_key}_top{top_k}.csv"
                        write_csv_rows(
                            variant_path,
                            build_artifact_risk_score_rows(
                                clue_rows=clue_rows,
                                candidate_rows=bridge_rows,
                                risk_head=risk_head,
                                top_k=top_k,
                                risk_weight=0.25,
                                boundary_fraction=boundary_fraction,
                            ),
                        )
                        bridge_paths[(weight, variant_key, int(top_k))] = variant_path
                    continue
                if _is_constrained_support_variant(variant_key):
                    for top_k in top_k_values:
                        stats_core_fraction, env_dominant_cap = _constrained_variant_params(variant_key, int(top_k))
                        variant_path = out_dir / f"scores_{source}_{_weight_label(weight)}_{variant_key}_top{top_k}.csv"
                        write_csv_rows(
                            variant_path,
                            build_constrained_support_score_rows(
                                clue_rows=clue_rows,
                                stats_rows=stats_rows,
                                candidate_rows=bridge_rows,
                                top_k=top_k,
                                stats_core_fraction=stats_core_fraction,
                                env_dominant_cap=env_dominant_cap,
                            ),
                        )
                        bridge_paths[(weight, variant_key, int(top_k))] = variant_path
                    continue
                variant_path = out_dir / f"scores_{source}_{_weight_label(weight)}_{variant_key}.csv"
                write_csv_rows(
                    variant_path,
                    build_support_variant_score_rows(
                        clue_rows=clue_rows,
                        stats_rows=stats_rows,
                        bridge_rows=bridge_rows,
                        variant=variant_key,
                    ),
                )
                bridge_paths[(weight, variant_key)] = variant_path

    baseline_base = with_runtime_overrides(
        with_dataset_path(load_config(baseline_config_path), dataset_path),
        official_dfr_num_retrains=num_retrains,
        training_device=training_device,
    )
    candidate_base = with_runtime_overrides(
        with_dataset_path(load_config(candidate_config_path), dataset_path),
        official_dfr_num_retrains=num_retrains,
        training_device=training_device,
    )
    rows: list[dict[str, str]] = []
    baseline_by_seed: dict[int, dict[str, float]] = {}
    stats_by_seed_top_k: dict[tuple[int, int], dict[str, float]] = {}

    with output_csv.open("w", encoding="utf-8", newline="") as handle, tempfile.TemporaryDirectory() as tmp_dir:
        writer: csv.DictWriter[str] | None = None

        def write_row(row: dict[str, str]) -> None:
            nonlocal writer
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

        tmp_root = Path(tmp_dir)
        for seed in seeds:
            baseline = copy.deepcopy(baseline_base)
            baseline["seed"] = seed
            baseline["name"] = f"{baseline.get('name', 'official_dfr')}_seed{seed}"
            baseline_path = tmp_root / f"{baseline['name']}.yaml"
            baseline_path.write_text(yaml.safe_dump(baseline, sort_keys=False), encoding="utf-8")
            baseline_run = run_experiment(baseline_path, output_root)
            baseline_metrics = _metric_payload(baseline_run)
            baseline_by_seed[seed] = baseline_metrics
            write_row(
                _row(
                    row_type="baseline",
                    label="official_dfr",
                    seed=seed,
                    top_k=0,
                    weight=None,
                    run_dir=baseline_run,
                    metrics=baseline_metrics,
                    baseline_metrics=None,
                    stats_metrics=None,
                )
            )

        for seed, top_k in itertools.product(seeds, top_k_values):
            stats_config = build_downstream_candidate(
                {**copy.deepcopy(candidate_base), "seed": seed},
                label="stats",
                top_k=top_k,
                score_path=stats_path,
                prune_soft_scores=True,
            )
            stats_config["name"] = f"{stats_config['name']}_seed{seed}"
            stats_path_tmp = tmp_root / f"{stats_config['name']}.yaml"
            stats_path_tmp.write_text(yaml.safe_dump(stats_config, sort_keys=False), encoding="utf-8")
            stats_run = run_experiment(stats_path_tmp, output_root)
            stats_metrics = _metric_payload(stats_run)
            stats_by_seed_top_k[(seed, top_k)] = stats_metrics
            write_row(
                _row(
                    row_type="stats_control",
                    label="stats",
                    seed=seed,
                    top_k=top_k,
                    weight=None,
                    run_dir=stats_run,
                    metrics=stats_metrics,
                    baseline_metrics=baseline_by_seed[seed],
                    stats_metrics=None,
                )
            )

        for seed, top_k, control_index in itertools.product(seeds, top_k_values, sorted(random_paths)):
            label = f"random_score_{control_index}"
            random_config = build_downstream_candidate(
                {**copy.deepcopy(candidate_base), "seed": seed},
                label="bridge_fused",
                top_k=top_k,
                score_path=random_paths[control_index],
                prune_soft_scores=True,
            )
            random_config["name"] = f"{candidate_base.get('name', 'bridge_fused')}_{label}_top{top_k}_seed{seed}"
            random_path_tmp = tmp_root / f"{random_config['name']}.yaml"
            random_path_tmp.write_text(yaml.safe_dump(random_config, sort_keys=False), encoding="utf-8")
            random_run = run_experiment(random_path_tmp, output_root)
            random_metrics = _metric_payload(random_run)
            write_row(
                _row(
                    row_type="random_control",
                    label=f"{label}_top{top_k}",
                    seed=seed,
                    top_k=top_k,
                    weight=None,
                    run_dir=random_run,
                    metrics=random_metrics,
                    baseline_metrics=baseline_by_seed[seed],
                    stats_metrics=stats_by_seed_top_k[(seed, top_k)],
                )
            )

        candidate_items: list[tuple[float, str, Path, float | None, int | None]] = []
        for weight in bridge_fused_weights:
            candidate_items.append((weight, f"{source}_{_weight_label(weight)}", bridge_paths[weight], weight, None))
            if source == "bridge_fused":
                for variant in support_variants:
                    variant_key = variant.strip().lower()
                    if (
                        _is_constrained_support_variant(variant_key)
                        or _is_artifact_risk_variant(variant_key)
                        or _is_active_boundary_variant(variant_key)
                    ):
                        for top_k in top_k_values:
                            candidate_items.append(
                                (
                                    weight,
                                    f"{source}_{_weight_label(weight)}_{variant_key}",
                                    bridge_paths[(weight, variant_key, int(top_k))],
                                    weight,
                                    int(top_k),
                                )
                            )
                    else:
                        candidate_items.append(
                            (
                                weight,
                                f"{source}_{_weight_label(weight)}_{variant_key}",
                                bridge_paths[(weight, variant_key)],
                                weight,
                                None,
                            )
                        )

        for seed, top_k, candidate_item in itertools.product(seeds, top_k_values, candidate_items):
            _weight, label, score_path, row_weight, item_top_k = candidate_item
            if item_top_k is not None and item_top_k != top_k:
                continue
            config = build_downstream_candidate(
                {**copy.deepcopy(candidate_base), "seed": seed},
                label=source,
                top_k=top_k,
                score_path=score_path,
                prune_soft_scores=True,
            )
            config["name"] = f"{candidate_base.get('name', 'bridge_fused')}_{label}_top{top_k}_seed{seed}"
            config_path = tmp_root / f"{config['name']}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            run_dir = run_experiment(config_path, output_root)
            metrics = _metric_payload(run_dir)
            write_row(
                _row(
                    row_type="candidate",
                    label=f"{label}_top{top_k}",
                    seed=seed,
                    top_k=top_k,
                    weight=row_weight,
                    run_dir=run_dir,
                    metrics=metrics,
                    baseline_metrics=baseline_by_seed[seed],
                    stats_metrics=stats_by_seed_top_k[(seed, top_k)],
                )
            )

    summary = _summary(rows)
    summary.update(
        {
            "output_csv": str(output_csv),
            "out_dir": str(out_dir),
            "seeds": seeds,
            "top_k_values": top_k_values,
            "bridge_fused_weights": bridge_fused_weights,
            "support_variants": support_variants,
            "bridge_score_source": source,
            "random_control_count": random_control_count,
            "num_retrains": num_retrains,
            "bridge_alpha": bridge_alpha,
            "bridge_exclude_datasets": bridge_exclude_datasets,
            "policy_input_dir": str(policy_input_dir),
            "policy_alpha": policy_alpha,
            "policy_exclude_datasets": policy_exclude_datasets,
        }
    )
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", default="configs/benchmarks/waterbirds_features_official_dfr_val_tr_retrains50.yaml")
    parser.add_argument("--candidate-config", default="configs/benchmarks/waterbirds_features_official_causal_shrink_dfr_val_tr_gentle_retrains50.yaml")
    parser.add_argument("--dataset-path", default="data/waterbirds/features_official_erm_official_repro.csv")
    parser.add_argument("--bridge-input-dir", default="outputs/dfr_sweeps/llm_clue_fixture_experiments")
    parser.add_argument("--out-dir", default="outputs/dfr_sweeps/bridge_fused_sweep")
    parser.add_argument("--output-csv", default="outputs/dfr_sweeps/bridge-fused-sweep.csv")
    parser.add_argument("--output-json", default="outputs/dfr_sweeps/bridge-fused-sweep.json")
    parser.add_argument("--seeds", nargs="*")
    parser.add_argument("--top-k", nargs="*")
    parser.add_argument("--bridge-fused-weights", nargs="*")
    parser.add_argument("--support-variant", action="append", default=[])
    parser.add_argument(
        "--bridge-score-source",
        choices=["bridge_fused", "bridge_gated", "pairwise_bridge_fused", "policy_fused", "policy_safe"],
        default="bridge_fused",
    )
    parser.add_argument("--bridge-alpha", type=float, default=10.0)
    parser.add_argument("--bridge-exclude-dataset", action="append", default=["waterbirds"])
    parser.add_argument("--policy-input-dir", default="outputs/dfr_sweeps/llm_clue_fixture_experiments_20260502_refreshed")
    parser.add_argument("--policy-alpha", type=float, default=10.0)
    parser.add_argument("--policy-exclude-dataset", action="append", default=["waterbirds"])
    parser.add_argument("--card-top-k", type=int, default=16)
    parser.add_argument("--random-control-count", type=int, default=0)
    parser.add_argument("--num-retrains", type=int, default=5)
    parser.add_argument("--training-device", default="")
    parser.add_argument("--output-root", default="outputs/runs")
    args = parser.parse_args()

    summary = run_bridge_fused_sweep(
        baseline_config_path=Path(args.baseline_config),
        candidate_config_path=Path(args.candidate_config),
        dataset_path=args.dataset_path,
        bridge_input_dir=Path(args.bridge_input_dir),
        out_dir=Path(args.out_dir),
        output_csv=Path(args.output_csv),
        output_json=Path(args.output_json),
        seeds=_int_values(args.seeds, [101]),
        top_k_values=_int_values(args.top_k, [384, 512, 640]),
        bridge_fused_weights=_float_values(args.bridge_fused_weights, [0.1, 0.2, 0.3]),
        support_variants=list(dict.fromkeys(args.support_variant)),
        bridge_score_source=str(args.bridge_score_source),
        bridge_alpha=float(args.bridge_alpha),
        bridge_exclude_datasets=list(dict.fromkeys(args.bridge_exclude_dataset)),
        policy_input_dir=Path(args.policy_input_dir),
        policy_alpha=float(args.policy_alpha),
        policy_exclude_datasets=list(dict.fromkeys(args.policy_exclude_dataset)),
        card_top_k=int(args.card_top_k),
        random_control_count=int(args.random_control_count),
        num_retrains=int(args.num_retrains),
        training_device=args.training_device or None,
        output_root=Path(args.output_root),
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
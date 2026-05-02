from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .llm_clue_planner import ACTION_CATALOG


REWARD_SCHEMA_VERSION = "rl_clue_reward/v1"

POLICY_FEATURE_COLUMNS = (
    "label_corr",
    "env_corr",
    "corr_margin",
    "abs_corr_margin",
    "uncertainty",
    "top_group_entropy",
    "label_env_disentanglement",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class OfflineCluePolicy:
    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    train_reward_count: int
    action_catalog: tuple[str, ...] = ACTION_CATALOG


def clue_reward_row(
    *,
    packet: Mapping[str, Any],
    trace: Mapping[str, Any] | None = None,
    feature_clue: Mapping[str, Any] | None = None,
    dataset: str = "",
    reward_scope: str = "fixture",
) -> dict[str, Any]:
    trace_row = dict(trace or {})
    clue_row = dict(feature_clue or {})
    action = str(trace_row.get("action", "")).strip().lower()
    if action and action not in ACTION_CATALOG:
        raise ValueError(f"Unknown action {action!r} in reward row.")
    test_reward = max(_safe_float(trace_row.get("test_value")), 0.0)
    score_delta_reward = max(_safe_float(trace_row.get("score_delta")), 0.0)
    control_pass_reward = 0.25 if _safe_bool(trace_row.get("passed_control")) else 0.0
    hypothesis_reward = 0.25 if _safe_bool(trace_row.get("hypothesis_correct")) else 0.0
    causal_target = _safe_float(clue_row.get("causal_target"), _safe_float(packet.get("causal_target")))
    if math.isnan(causal_target):
        causal_target = 0.0
    causal_target_reward = max(causal_target, 0.0)
    total_reward = test_reward + score_delta_reward + control_pass_reward + hypothesis_reward + causal_target_reward
    scope = str(reward_scope).strip().lower() or "fixture"
    trainable = scope not in {"benchmark_final", "waterbirds_test", "test_wga"}
    row: dict[str, Any] = {
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "dataset": dataset or str(packet.get("dataset", "")),
        "reward_scope": scope,
        "trainable_reward": trainable,
        "candidate_id": str(packet.get("candidate_id", trace_row.get("candidate_id", ""))),
        "feature_name": str(packet.get("feature_name", trace_row.get("feature_name", ""))),
        "feature_index": str(packet.get("feature_index", trace_row.get("feature_index", ""))),
        "packet_hash": str(packet.get("packet_hash", trace_row.get("packet_hash", ""))),
        "action": action,
        "action_index": ACTION_CATALOG.index(action) if action in ACTION_CATALOG else -1,
        "test_reward": test_reward,
        "score_delta_reward": score_delta_reward,
        "control_pass_reward": control_pass_reward,
        "hypothesis_reward": hypothesis_reward,
        "causal_target_reward": causal_target_reward,
        "total_reward": total_reward,
    }
    for column in POLICY_FEATURE_COLUMNS:
        row[column] = _safe_float(packet.get(column, trace_row.get(column)))
    return row


def build_clue_reward_rows(
    *,
    packets: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
    feature_clues: Mapping[str, Mapping[str, Any]] | None = None,
    dataset: str = "",
    reward_scope: str = "fixture",
) -> list[dict[str, Any]]:
    packets_by_id = {str(packet.get("candidate_id", "")): packet for packet in packets}
    packets_by_feature = {str(packet.get("feature_name", "")): packet for packet in packets}
    clues = dict(feature_clues or {})
    rows: list[dict[str, Any]] = []
    for trace in traces:
        packet = packets_by_id.get(str(trace.get("candidate_id", "")))
        if packet is None:
            packet = packets_by_feature.get(str(trace.get("feature_name", "")))
        if packet is None:
            continue
        feature_name = str(packet.get("feature_name", trace.get("feature_name", "")))
        rows.append(
            clue_reward_row(
                packet=packet,
                trace=trace,
                feature_clue=clues.get(feature_name, {}),
                dataset=dataset,
                reward_scope=reward_scope,
            )
        )
    return rows


def assert_no_benchmark_final_training(rows: Sequence[Mapping[str, Any]]) -> None:
    leaked = [row for row in rows if not _safe_bool(row.get("trainable_reward", True))]
    if leaked:
        raise ValueError("Benchmark-final/test reward rows cannot be used for policy training.")


def _policy_feature_vector(row: Mapping[str, Any], *, action: str | None = None) -> list[float]:
    values = [_safe_float(row.get(column)) for column in POLICY_FEATURE_COLUMNS]
    values.append(_safe_float(row.get("label_corr")) - _safe_float(row.get("env_corr")))
    selected_action = (action or str(row.get("action", ""))).strip().lower()
    values.extend(1.0 if selected_action == catalog_action else 0.0 for catalog_action in ACTION_CATALOG)
    values.append(1.0)
    return values


def train_offline_clue_policy(rows: Sequence[Mapping[str, Any]], *, alpha: float = 1.0) -> OfflineCluePolicy:
    assert_no_benchmark_final_training(rows)
    train_rows = [dict(row) for row in rows if _safe_bool(row.get("trainable_reward", True))]
    if not train_rows:
        raise ValueError("No trainable reward rows found.")
    x = np.asarray([_policy_feature_vector(row) for row in train_rows], dtype=np.float64)
    y = np.asarray([_safe_float(row.get("total_reward")) for row in train_rows], dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    xz = (x - mean) / scale
    penalty = float(alpha) * np.eye(xz.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    weights = np.linalg.pinv(xz.T @ xz + penalty) @ xz.T @ y
    return OfflineCluePolicy(weights=weights, mean=mean, scale=scale, train_reward_count=len(train_rows))


def predict_action_value(row: Mapping[str, Any], policy: OfflineCluePolicy, *, action: str) -> float:
    x = np.asarray(_policy_feature_vector(row, action=action), dtype=np.float64)
    return float(((x - policy.mean) / policy.scale) @ policy.weights)


def score_policy_packets(packets: Sequence[Mapping[str, Any]], policy: OfflineCluePolicy) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for packet in packets:
        values = {action: predict_action_value(packet, policy, action=action) for action in policy.action_catalog}
        best_action = max(values, key=values.get)
        score = values[best_action]
        rows.append(
            {
                "dataset": str(packet.get("dataset", "")),
                "feature_index": str(packet.get("feature_index", "")),
                "feature_name": str(packet.get("feature_name", "")),
                "support_score": f"{score:.6f}",
                "rank_score": f"{score:.6f}",
                "score": f"{score:.6f}",
                "score_source": "offline_clue_policy",
                "best_action": best_action,
            }
        )
    return rows
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from .llm_clue_planner import ACTION_CATALOG, HYPOTHESIS_TYPES, ClueHypothesis, ClueTestSpec


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def hypothesis_label_from_packet(packet: Mapping[str, Any]) -> str:
    label_corr = _safe_float(packet.get("label_corr"))
    env_corr = _safe_float(packet.get("env_corr"))
    alignment = str(packet.get("activation_alignment", "mixed")).strip().lower()
    if label_corr > env_corr + 0.05 and alignment == "label":
        return "causal"
    if env_corr > label_corr + 0.05 and alignment == "environment":
        return "nuisance"
    if abs(label_corr - env_corr) <= 0.05 or alignment == "mixed":
        return "mixed"
    return "uncertain"


def action_value_from_result(result: Mapping[str, Any]) -> float:
    passed = bool(result.get("test_passed_control", False))
    label_effect = _safe_float(result.get("test_effect_label_delta"))
    env_effect = abs(_safe_float(result.get("test_effect_env_delta")))
    selectivity = _safe_float(result.get("test_effect_selectivity"))
    random_delta = abs(_safe_float(result.get("test_random_control_delta")))
    cost = max(_safe_float(result.get("test_cost", 1.0), default=1.0), 1e-6)
    evidence = max(label_effect + selectivity - env_effect - random_delta, 0.0)
    if passed:
        evidence += 0.25
    return evidence / cost


def score_delta_from_result(result: Mapping[str, Any]) -> float:
    explicit = result.get("score_delta")
    if explicit not in (None, ""):
        return _safe_float(explicit)
    return action_value_from_result(result) - abs(_safe_float(result.get("test_random_control_delta")))


def _packet_features(packet: Mapping[str, Any]) -> dict[str, float]:
    label_corr = _safe_float(packet.get("label_corr"))
    env_corr = _safe_float(packet.get("env_corr"))
    corr_margin = _safe_float(packet.get("corr_margin"), default=label_corr - env_corr)
    return {
        "label_corr": label_corr,
        "env_corr": env_corr,
        "corr_margin": corr_margin,
        "abs_corr_margin": abs(corr_margin),
        "uncertainty": _safe_float(packet.get("uncertainty")),
        "top_group_entropy": _safe_float(packet.get("top_group_entropy")),
        "label_env_disentanglement": _safe_float(packet.get("label_env_disentanglement")),
    }


def bridge_training_row(
    *,
    packet: Mapping[str, Any],
    hypothesis: ClueHypothesis,
    test_spec: ClueTestSpec,
    result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target_label = hypothesis_label_from_packet(packet)
    result_row = dict(result or {})
    action_value = action_value_from_result(result_row) if result is not None else 0.0
    score_delta = score_delta_from_result(result_row) if result is not None else 0.0
    row: dict[str, Any] = {
        "candidate_id": packet.get("candidate_id", hypothesis.candidate_id),
        "feature_name": packet.get("feature_name", hypothesis.feature_name),
        "packet_hash": packet.get("packet_hash", ""),
        "proposed_hypothesis_label": hypothesis.hypothesis_type,
        "target_hypothesis_label": target_label,
        "hypothesis_correct": hypothesis.hypothesis_type == target_label,
        "action": test_spec.action,
        "action_index": ACTION_CATALOG.index(test_spec.action) if test_spec.action in ACTION_CATALOG else -1,
        "reason_code": test_spec.reason_code,
        "test_value": action_value,
        "score_delta": score_delta,
        "passed_control": bool(result_row.get("test_passed_control", False)),
    }
    row.update(_packet_features(packet))
    for hypothesis_type in HYPOTHESIS_TYPES:
        row[f"target_is_{hypothesis_type}"] = 1.0 if target_label == hypothesis_type else 0.0
    return row


def build_bridge_training_rows(
    packets: Sequence[Mapping[str, Any]],
    hypotheses: Sequence[ClueHypothesis],
    tests: Sequence[ClueTestSpec],
    results: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    packets_by_id = {str(packet.get("candidate_id", "")): packet for packet in packets}
    hypothesis_by_id = {hypothesis.candidate_id: hypothesis for hypothesis in hypotheses}
    results_by_id = {str(row.get("candidate_id", "")): row for row in (results or [])}
    rows: list[dict[str, Any]] = []
    for test_spec in tests:
        packet = packets_by_id.get(test_spec.candidate_id)
        hypothesis = hypothesis_by_id.get(test_spec.candidate_id)
        if packet is None or hypothesis is None:
            continue
        rows.append(
            bridge_training_row(
                packet=packet,
                hypothesis=hypothesis,
                test_spec=test_spec,
                result=results_by_id.get(test_spec.candidate_id),
            )
        )
    return rows

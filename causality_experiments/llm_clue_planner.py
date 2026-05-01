from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from typing import Any, Protocol


ACTION_CATALOG = (
    "feature_mean_ablation",
    "feature_zero_ablation",
    "feature_shrink",
    "donor_swap_same_label_diff_env",
    "donor_swap_diff_label_same_env",
    "conditional_signal_check",
    "probe_selectivity_check",
)

HYPOTHESIS_TYPES = ("causal", "nuisance", "mixed", "artifact", "redundant", "uncertain")


@dataclass(frozen=True)
class ClueHypothesis:
    candidate_id: str
    feature_name: str
    hypothesis_type: str
    confidence: float
    reason_code: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ClueTestSpec:
    candidate_id: str
    feature_name: str
    action: str
    expected_direction: str
    control: str
    cost: float
    reason_code: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class CluePlan:
    hypotheses: tuple[ClueHypothesis, ...]
    tests: tuple[ClueTestSpec, ...]
    backend: str
    repaired: bool = False
    fallback: bool = False


class CluePlannerBackend(Protocol):
    name: str

    def complete(self, prompt: str) -> str:
        ...


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _clamp01(value: Any) -> float:
    return min(max(_safe_float(value), 0.0), 1.0)


def _extract_json_object(text: str) -> tuple[dict[str, Any], bool]:
    stripped = text.strip()
    try:
        return json.loads(stripped), False
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(stripped[start : end + 1]), True


def _normalise_hypothesis(row: Mapping[str, Any]) -> ClueHypothesis:
    hypothesis_type = str(row.get("hypothesis_type", "uncertain")).strip().lower()
    if hypothesis_type not in HYPOTHESIS_TYPES:
        hypothesis_type = "uncertain"
    candidate_id = str(row.get("candidate_id", "")).strip()
    feature_name = str(row.get("feature_name", "")).strip()
    if not candidate_id or not feature_name:
        raise ValueError("Each hypothesis requires candidate_id and feature_name.")
    evidence = row.get("evidence_ids", ())
    if isinstance(evidence, str):
        evidence_ids = (evidence,)
    else:
        evidence_ids = tuple(str(item) for item in evidence or ())
    return ClueHypothesis(
        candidate_id=candidate_id,
        feature_name=feature_name,
        hypothesis_type=hypothesis_type,
        confidence=_clamp01(row.get("confidence", 0.0)),
        reason_code=str(row.get("reason_code", "unspecified")).strip() or "unspecified",
        evidence_ids=evidence_ids,
    )


def _normalise_test(row: Mapping[str, Any]) -> ClueTestSpec:
    candidate_id = str(row.get("candidate_id", "")).strip()
    feature_name = str(row.get("feature_name", "")).strip()
    action = str(row.get("action", "")).strip().lower()
    if not candidate_id or not feature_name:
        raise ValueError("Each test spec requires candidate_id and feature_name.")
    if action not in ACTION_CATALOG:
        raise ValueError(f"Unknown test action {action!r}.")
    evidence = row.get("evidence_ids", ())
    if isinstance(evidence, str):
        evidence_ids = (evidence,)
    else:
        evidence_ids = tuple(str(item) for item in evidence or ())
    return ClueTestSpec(
        candidate_id=candidate_id,
        feature_name=feature_name,
        action=action,
        expected_direction=str(row.get("expected_direction", "unknown")).strip() or "unknown",
        control=str(row.get("control", "random_feature")).strip() or "random_feature",
        cost=max(_safe_float(row.get("cost", 1.0), default=1.0), 0.0),
        reason_code=str(row.get("reason_code", "unspecified")).strip() or "unspecified",
        evidence_ids=evidence_ids,
    )


def parse_clue_plan(text: str, *, backend: str) -> CluePlan:
    payload, repaired = _extract_json_object(text)
    hypotheses = tuple(_normalise_hypothesis(row) for row in payload.get("hypotheses", []))
    tests = tuple(_normalise_test(row) for row in payload.get("tests", []))
    if not hypotheses:
        raise ValueError("Planner output must include at least one hypothesis.")
    if not tests:
        raise ValueError("Planner output must include at least one test spec.")
    return CluePlan(hypotheses=hypotheses, tests=tests, backend=backend, repaired=repaired)


def render_planner_prompt(packets: Sequence[Mapping[str, Any]], *, max_packets: int = 16) -> str:
    compact_packets = []
    for packet in packets[: max(0, int(max_packets))]:
        compact_packets.append(
            {
                "candidate_id": packet.get("candidate_id", ""),
                "feature_name": packet.get("feature_name", ""),
                "feature_group": packet.get("feature_group", ""),
                "label_corr": packet.get("label_corr", 0.0),
                "env_corr": packet.get("env_corr", 0.0),
                "corr_margin": packet.get("corr_margin", 0.0),
                "activation_alignment": packet.get("activation_alignment", ""),
                "uncertainty": packet.get("uncertainty", 0.0),
                "packet_hash": packet.get("packet_hash", ""),
            }
        )
    payload = {
        "instruction": "Return JSON with hypotheses and tests over the provided latent clue packets.",
        "hypothesis_types": list(HYPOTHESIS_TYPES),
        "action_catalog": list(ACTION_CATALOG),
        "packets": compact_packets,
    }
    return json.dumps(payload, sort_keys=True)


class MockCluePlannerBackend:
    name = "mock"

    def complete(self, prompt: str) -> str:
        payload = json.loads(prompt)
        packets = sorted(
            payload.get("packets", []),
            key=lambda packet: (
                _safe_float(packet.get("label_corr")) - _safe_float(packet.get("env_corr")),
                abs(_safe_float(packet.get("corr_margin"))),
                -_safe_float(packet.get("uncertainty")),
            ),
            reverse=True,
        )
        hypotheses: list[dict[str, Any]] = []
        tests: list[dict[str, Any]] = []
        for packet in packets[:3]:
            label_corr = _safe_float(packet.get("label_corr"))
            env_corr = _safe_float(packet.get("env_corr"))
            margin = label_corr - env_corr
            if margin > 0.05:
                hypothesis_type = "causal"
                action = "feature_mean_ablation"
                direction = "label_logit_drop"
                reason = "label_over_env"
            elif margin < -0.05:
                hypothesis_type = "nuisance"
                action = "donor_swap_same_label_diff_env"
                direction = "env_effect_drop"
                reason = "env_over_label"
            else:
                hypothesis_type = "mixed"
                action = "conditional_signal_check"
                direction = "disambiguate_label_env"
                reason = "ambiguous_signal"
            candidate_id = str(packet.get("candidate_id", ""))
            feature_name = str(packet.get("feature_name", ""))
            evidence_ids = [str(packet.get("packet_hash", candidate_id))]
            hypotheses.append(
                {
                    "candidate_id": candidate_id,
                    "feature_name": feature_name,
                    "hypothesis_type": hypothesis_type,
                    "confidence": min(abs(margin) + 0.5, 1.0),
                    "reason_code": reason,
                    "evidence_ids": evidence_ids,
                }
            )
            tests.append(
                {
                    "candidate_id": candidate_id,
                    "feature_name": feature_name,
                    "action": action,
                    "expected_direction": direction,
                    "control": "random_feature",
                    "cost": 1.0,
                    "reason_code": reason,
                    "evidence_ids": evidence_ids,
                }
            )
        return json.dumps({"hypotheses": hypotheses, "tests": tests}, sort_keys=True)


def plan_from_backend(
    packets: Sequence[Mapping[str, Any]],
    backend: CluePlannerBackend,
    *,
    max_packets: int = 16,
) -> CluePlan:
    prompt = render_planner_prompt(packets, max_packets=max_packets)
    try:
        return parse_clue_plan(backend.complete(prompt), backend=backend.name)
    except Exception:
        fallback_backend = MockCluePlannerBackend()
        fallback = parse_clue_plan(fallback_backend.complete(prompt), backend=fallback_backend.name)
        return CluePlan(
            hypotheses=fallback.hypotheses,
            tests=fallback.tests,
            backend=fallback.backend,
            repaired=fallback.repaired,
            fallback=True,
        )

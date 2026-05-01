from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

from .clues import build_feature_cards
from .data import DatasetBundle
from .discovery import build_feature_clue_rows


LATENT_CLUE_PACKET_SCHEMA_VERSION = "latent_clue_packet/v1"


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _feature_group(feature_name: str) -> str:
    if feature_name.startswith("feature_"):
        parts = feature_name.split("_")
        if len(parts) >= 3 and not parts[1].isdigit():
            return parts[1]
    if "background" in feature_name.lower():
        return "background"
    if any(token in feature_name.lower() for token in ("bird", "shape", "label")):
        return "label"
    return "generic"


def _stable_payload(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in packet.items() if key != "packet_hash"}


def packet_hash(packet: Mapping[str, Any]) -> str:
    payload = json.dumps(_stable_payload(packet), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _uncertainty_from_scores(label_corr: float, env_corr: float, alignment: str) -> float:
    gap_uncertainty = 1.0 - min(abs(label_corr - env_corr), 1.0)
    if alignment == "mixed":
        return max(gap_uncertainty, 0.75)
    return min(max(gap_uncertainty, 0.0), 1.0)


def build_latent_clue_packets(
    bundle: DatasetBundle,
    *,
    split_name: str = "train",
    top_k: int = 8,
    max_packets: int | None = None,
    probe_summary: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cards = build_feature_cards(bundle, split_name=split_name, top_k=top_k)
    clue_rows = build_feature_clue_rows(bundle, split_name=split_name)
    clues_by_name = {str(row["feature_name"]): row for row in clue_rows}
    packets: list[dict[str, Any]] = []
    for card in cards:
        feature_name = str(card["feature_name"])
        clue = clues_by_name.get(feature_name, {})
        label_corr = _safe_float(clue.get("label_corr", card.get("label_corr")))
        env_corr = _safe_float(clue.get("env_corr", card.get("env_corr")))
        corr_margin = _safe_float(clue.get("corr_margin", card.get("corr_margin")))
        alignment = str(card.get("activation_alignment", "mixed"))
        feature_index = int(card.get("feature_index", clue.get("feature_index", 0)))
        packet: dict[str, Any] = {
            "schema_version": LATENT_CLUE_PACKET_SCHEMA_VERSION,
            "dataset": bundle.name,
            "split": split_name,
            "candidate_id": f"{bundle.name}:{split_name}:{feature_name}",
            "feature_index": feature_index,
            "feature_name": feature_name,
            "feature_group": _feature_group(feature_name),
            "task": bundle.task,
            "modality": str((bundle.metadata or {}).get("modality", "features")),
            "label_corr": label_corr,
            "env_corr": env_corr,
            "corr_margin": corr_margin,
            "abs_corr_margin": abs(corr_margin),
            "activation_alignment": alignment,
            "activation_label_gap": _safe_float(card.get("activation_label_gap")),
            "activation_env_gap": _safe_float(card.get("activation_env_gap")),
            "top_group_entropy": _safe_float(card.get("top_group_entropy")),
            "label_env_disentanglement": _safe_float(card.get("label_env_disentanglement")),
            "has_ground_truth_mask": bool(clue.get("has_ground_truth_mask", False)),
            "causal_target": _safe_float(clue.get("causal_target"), default=float("nan")),
            "uncertainty": _uncertainty_from_scores(label_corr, env_corr, alignment),
            "cost_to_test": 1.0,
            "feature_statement": str(card.get("feature_statement", "")),
            "probe_summary": dict(probe_summary or {}),
        }
        packet["packet_hash"] = packet_hash(packet)
        packets.append(packet)
    packets.sort(
        key=lambda row: (
            _safe_float(row.get("uncertainty")),
            _safe_float(row.get("abs_corr_margin")),
            str(row.get("candidate_id", "")),
        ),
        reverse=True,
    )
    if max_packets is not None:
        packets = packets[: max(0, int(max_packets))]
    return packets


def packets_to_jsonl(packets: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(json.dumps(dict(packet), sort_keys=True) for packet in packets)

from __future__ import annotations

from collections import Counter
import csv
import math
from pathlib import Path
import re
from typing import Any

import torch

from .data import DatasetBundle
from .discovery import build_feature_clue_rows


WATERBIRDS_CAUSAL_TERMS = {
    "beak",
    "bill",
    "bird",
    "body",
    "feather",
    "label",
    "landbird",
    "shape",
    "waterbird",
    "wing",
}

WATERBIRDS_SPURIOUS_TERMS = {
    "background",
    "environment",
    "forest",
    "habitat",
    "land",
    "lake",
    "ocean",
    "place",
    "shore",
    "tree",
    "water",
}

GENERAL_CAUSAL_TERMS = {
    "causal",
    "cause",
    "class",
    "label",
    "object",
    "shape",
    "target",
}

GENERAL_SPURIOUS_TERMS = {
    "background",
    "confounder",
    "environment",
    "nuisance",
    "place",
    "shortcut",
    "spurious",
}


def _safe_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.float().mean().item())


def _normalised_entropy(values: torch.Tensor, *, support_size: int) -> float:
    if values.numel() == 0 or support_size <= 1:
        return 0.0
    counts = Counter(int(value) for value in values.detach().cpu().view(-1).tolist())
    total = float(sum(counts.values()))
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return float(entropy / math.log(max(support_size, 2)))


def _mode(values: torch.Tensor) -> int:
    if values.numel() == 0:
        return 0
    counts = Counter(int(value) for value in values.detach().cpu().view(-1).tolist())
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _activation_alignment(label_gap: float, env_gap: float, *, tolerance: float = 0.05) -> str:
    label_strength = abs(label_gap)
    env_strength = abs(env_gap)
    if label_strength > env_strength + tolerance:
        return "label"
    if env_strength > label_strength + tolerance:
        return "environment"
    return "mixed"


def _feature_statement(feature_name: str, alignment: str, label_gap: float, env_gap: float) -> str:
    if alignment == "label":
        focus = "label-aligned"
    elif alignment == "environment":
        focus = "environment-aligned"
    else:
        focus = "mixed label and environment"
    return (
        f"{feature_name} top activations are {focus}; "
        f"label activation gap {label_gap:.3f}; environment activation gap {env_gap:.3f}."
    )


def build_feature_cards(
    bundle: DatasetBundle,
    *,
    split_name: str = "train",
    top_k: int = 8,
) -> list[dict[str, Any]]:
    split = bundle.split(split_name)
    x = split["x"]
    y = split["y"].float()
    env = split["env"].float()
    group = split["group"].long()
    if x.shape[0] == 0:
        raise ValueError(f"Dataset {bundle.name!r} split {split_name!r} is empty.")
    count = max(1, min(int(top_k), int(x.shape[0])))
    group_support_size = max(int(torch.unique(group).numel()), 2)
    base_rows = build_feature_clue_rows(bundle, split_name=split_name)

    cards: list[dict[str, Any]] = []
    for row in base_rows:
        feature_index = int(row["feature_index"])
        feature_name = str(row["feature_name"])
        values = x[:, feature_index].float()
        top_indices = torch.topk(values, k=count).indices
        bottom_indices = torch.topk(-values, k=count).indices
        top_label_rate = _safe_mean(y[top_indices])
        top_env_rate = _safe_mean(env[top_indices])
        bottom_label_rate = _safe_mean(y[bottom_indices])
        bottom_env_rate = _safe_mean(env[bottom_indices])
        label_gap = top_label_rate - bottom_label_rate
        env_gap = top_env_rate - bottom_env_rate
        alignment = _activation_alignment(label_gap, env_gap)
        group_entropy = _normalised_entropy(group[top_indices], support_size=group_support_size)
        card = {
            "dataset": row["dataset"],
            "split": split_name,
            "feature_index": feature_index,
            "feature_name": feature_name,
            "feature_card_id": f"{row['dataset']}:{split_name}:{feature_name}",
            "task": row["task"],
            "modality": row["modality"],
            "label_corr": row["label_corr"],
            "env_corr": row["env_corr"],
            "corr_margin": row["corr_margin"],
            "mean": row["mean"],
            "std": row["std"],
            "top_activation_count": count,
            "bottom_activation_count": count,
            "top_label_rate": top_label_rate,
            "top_env_rate": top_env_rate,
            "bottom_label_rate": bottom_label_rate,
            "bottom_env_rate": bottom_env_rate,
            "activation_label_gap": label_gap,
            "activation_env_gap": env_gap,
            "activation_alignment": alignment,
            "top_group_entropy": group_entropy,
            "top_dominant_group": _mode(group[top_indices]),
            "label_env_disentanglement": abs(float(row["corr_margin"])),
            "semantic_hint": alignment,
            "feature_statement": _feature_statement(feature_name, alignment, label_gap, env_gap),
        }
        cards.append(card)
    return cards


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))


def _domain_terms(domain: str, dataset: str) -> tuple[set[str], set[str], str]:
    resolved = domain.strip().lower()
    if resolved == "auto":
        resolved = "waterbirds" if "waterbirds" in dataset.lower() else "general"
    if resolved == "waterbirds":
        return WATERBIRDS_CAUSAL_TERMS, WATERBIRDS_SPURIOUS_TERMS, resolved
    return GENERAL_CAUSAL_TERMS, GENERAL_SPURIOUS_TERMS, resolved


def _float_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return parsed


def build_language_clue_rows(
    feature_cards: list[dict[str, Any]],
    *,
    domain: str = "auto",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for card in feature_cards:
        dataset = str(card.get("dataset", ""))
        causal_terms, spurious_terms, resolved_domain = _domain_terms(domain, dataset)
        token_values = _tokens(str(card.get("feature_name", "")))
        causal_hits = sum(1 for token in token_values if token in causal_terms)
        spurious_hits = sum(1 for token in token_values if token in spurious_terms)
        alignment = str(card.get("activation_alignment", "")).strip().lower()
        label_strength = abs(_float_value(card, "activation_label_gap"))
        env_strength = abs(_float_value(card, "activation_env_gap"))
        activation_confidence = min(abs(label_strength - env_strength) * 2.0, 1.0)
        if alignment == "label":
            causal_hits += 2
        elif alignment == "environment":
            spurious_hits += 2
        elif alignment == "mixed":
            causal_hits += 1
            spurious_hits += 1

        evidence = float(causal_hits + spurious_hits)
        if evidence <= 0.0:
            causal_score = 0.0
            spurious_score = 0.0
            ambiguous_score = 1.0
            confidence = 0.0
        else:
            causal_score = causal_hits / evidence
            spurious_score = spurious_hits / evidence
            ambiguous_score = 1.0 - abs(causal_score - spurious_score)
            confidence = max(min(evidence / 4.0, 1.0), activation_confidence) * abs(causal_score - spurious_score)

        rows.append(
            {
                "dataset": dataset,
                "split": card.get("split", ""),
                "feature_index": card.get("feature_index", ""),
                "feature_name": card.get("feature_name", ""),
                "language_causal_score": f"{causal_score:.6f}",
                "language_spurious_score": f"{spurious_score:.6f}",
                "language_ambiguous_score": f"{ambiguous_score:.6f}",
                "language_confidence": f"{confidence:.6f}",
                "language_source_count": f"{evidence:.6f}",
                "language_prior_source": f"template:{resolved_domain}",
                "feature_card_path": card.get("feature_card_path", card.get("feature_card_id", "")),
                "top_activation_group_entropy": f"{_float_value(card, 'top_group_entropy'):.6f}",
                "label_env_disentanglement": f"{_float_value(card, 'label_env_disentanglement'):.6f}",
            }
        )
    return rows


def build_image_prototype_clue_rows(feature_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for card in feature_cards:
        label_strength = abs(_float_value(card, "activation_label_gap"))
        background_strength = abs(_float_value(card, "activation_env_gap"))
        total_strength = label_strength + background_strength
        if total_strength <= 1e-12:
            label_score = 0.5
            background_score = 0.5
            prompt_margin = 0.0
            confidence = 0.0
        else:
            label_score = label_strength / total_strength
            background_score = background_strength / total_strength
            prompt_margin = label_score - background_score
            separation = abs(prompt_margin)
            group_stability = 1.0 - min(max(_float_value(card, "top_group_entropy"), 0.0), 1.0)
            confidence = min(max(separation * (0.5 + 0.5 * group_stability), 0.0), 1.0)
        group_stability = 1.0 - min(max(_float_value(card, "top_group_entropy"), 0.0), 1.0)
        rows.append(
            {
                "dataset": card.get("dataset", ""),
                "split": card.get("split", ""),
                "feature_index": card.get("feature_index", ""),
                "feature_name": card.get("feature_name", ""),
                "image_label_score": f"{label_score:.6f}",
                "image_background_score": f"{background_score:.6f}",
                "image_group_stability": f"{group_stability:.6f}",
                "image_prompt_margin": f"{prompt_margin:.6f}",
                "image_confidence": f"{confidence:.6f}",
                "prototype_source_count": f"{float(card.get('top_activation_count', 0) or 0):.6f}",
                "feature_card_path": card.get("feature_card_path", card.get("feature_card_id", "")),
                "top_activation_group_entropy": f"{_float_value(card, 'top_group_entropy'):.6f}",
                "label_env_disentanglement": f"{_float_value(card, 'label_env_disentanglement'):.6f}",
            }
        )
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}.")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

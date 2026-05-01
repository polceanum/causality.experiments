from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch

from .data import DatasetBundle
from .llm_clue_planner import ClueTestSpec
from .methods import FittedModel


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _feature_names(bundle: DatasetBundle) -> list[str]:
    metadata = bundle.metadata or {}
    names = list(metadata.get("feature_columns", []))
    if len(names) == bundle.input_dim:
        return [str(name) for name in names]
    return [f"feature_{index}" for index in range(bundle.input_dim)]


def _feature_index(bundle: DatasetBundle, *, feature_name: str, packet: Mapping[str, Any] | None = None) -> int:
    names = _feature_names(bundle)
    if feature_name in names:
        return names.index(feature_name)
    if packet is not None and "feature_index" in packet:
        index = int(packet["feature_index"])
        if 0 <= index < bundle.input_dim:
            return index
    raise ValueError(f"Unknown feature {feature_name!r} for dataset {bundle.name!r}.")


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


def _edit_feature(
    values: torch.Tensor,
    *,
    feature_index: int,
    train_reference: torch.Tensor,
    action: str,
) -> torch.Tensor:
    edited = values.clone()
    if action == "feature_zero_ablation":
        edited[:, feature_index] = 0.0
    elif action == "feature_shrink":
        reference = train_reference[:, feature_index].mean()
        edited[:, feature_index] = 0.5 * values[:, feature_index] + 0.5 * reference
    else:
        reference = train_reference[:, feature_index].mean()
        edited[:, feature_index] = reference
    return edited


def _donor_edit(
    split: dict[str, torch.Tensor],
    *,
    feature_index: int,
    action: str,
) -> torch.Tensor:
    values = split["x"].clone()
    labels = split["y"].long()
    environments = split["env"].long()
    for row_index in range(values.shape[0]):
        if action == "donor_swap_diff_label_same_env":
            donor_mask = (labels != labels[row_index]) & (environments == environments[row_index])
        elif action == "donor_swap_same_label_diff_env":
            donor_mask = (labels == labels[row_index]) & (environments != environments[row_index])
        else:
            donor_mask = torch.ones_like(labels, dtype=torch.bool)
            donor_mask[row_index] = False
        donor_indices = torch.nonzero(donor_mask, as_tuple=False).flatten()
        if donor_indices.numel() == 0:
            continue
        donor_index = int(donor_indices[row_index % donor_indices.numel()].item())
        values[row_index, feature_index] = split["x"][donor_index, feature_index]
    return values


def _target_logit_drop(model: FittedModel, split: dict[str, torch.Tensor], edited_x: torch.Tensor) -> float:
    baseline_logits = model.predict(split["x"])
    edited_logits = model.predict(edited_x)
    labels = split["y"].long()
    row_indices = torch.arange(labels.shape[0], device=baseline_logits.device)
    baseline_targets = baseline_logits[row_indices, labels.to(baseline_logits.device)]
    edited_targets = edited_logits[row_indices, labels.to(edited_logits.device)]
    return float((baseline_targets - edited_targets).float().mean().item())


def _prediction_flip_rate(model: FittedModel, split: dict[str, torch.Tensor], edited_x: torch.Tensor) -> float:
    baseline = model.predict(split["x"]).argmax(dim=1)
    edited = model.predict(edited_x).argmax(dim=1)
    return float((baseline.cpu() != edited.cpu()).float().mean().item())


def _conditional_signal(split: dict[str, torch.Tensor], *, feature_index: int) -> tuple[float, float, float]:
    values = split["x"][:, feature_index]
    labels = split["y"].long()
    environments = split["env"].long()
    label_scores: list[float] = []
    env_scores: list[float] = []
    for environment in torch.unique(environments):
        mask = environments == environment
        if int(mask.sum()) > 1:
            label_scores.append(_safe_abs_corr(values[mask], labels[mask]))
    for label in torch.unique(labels):
        mask = labels == label
        if int(mask.sum()) > 1:
            env_scores.append(_safe_abs_corr(values[mask], environments[mask]))
    label_signal = sum(label_scores) / max(len(label_scores), 1)
    env_signal = sum(env_scores) / max(len(env_scores), 1)
    return label_signal, env_signal, label_signal - env_signal


def execute_clue_test(
    bundle: DatasetBundle,
    test_spec: ClueTestSpec,
    *,
    packet: Mapping[str, Any] | None = None,
    model: FittedModel | None = None,
    split_name: str = "test",
) -> dict[str, Any]:
    split = bundle.split(split_name)
    train = bundle.split("train")
    feature_index = _feature_index(bundle, feature_name=test_spec.feature_name, packet=packet)
    control_index = (feature_index + 1) % max(bundle.input_dim, 1)
    action = test_spec.action
    if action in {"donor_swap_same_label_diff_env", "donor_swap_diff_label_same_env"}:
        edited = _donor_edit(split, feature_index=feature_index, action=action)
        control_edited = _donor_edit(split, feature_index=control_index, action=action)
    elif action in {"feature_mean_ablation", "feature_zero_ablation", "feature_shrink", "probe_selectivity_check"}:
        edit_action = "feature_mean_ablation" if action == "probe_selectivity_check" else action
        edited = _edit_feature(split["x"], feature_index=feature_index, train_reference=train["x"], action=edit_action)
        control_edited = _edit_feature(split["x"], feature_index=control_index, train_reference=train["x"], action=edit_action)
    elif action == "conditional_signal_check":
        edited = split["x"]
        control_edited = split["x"]
    else:
        raise ValueError(f"Unsupported clue test action {action!r}.")

    label_signal, env_signal, selectivity = _conditional_signal(split, feature_index=feature_index)
    control_label_signal, control_env_signal, control_selectivity = _conditional_signal(split, feature_index=control_index)
    label_delta = 0.0
    random_delta = 0.0
    flip_rate = 0.0
    if model is not None and action != "conditional_signal_check":
        label_delta = _target_logit_drop(model, split, edited)
        random_delta = _target_logit_drop(model, split, control_edited)
        flip_rate = _prediction_flip_rate(model, split, edited)
    else:
        label_delta = label_signal
        random_delta = control_label_signal

    effect_margin = label_delta + selectivity - abs(env_signal) - abs(random_delta)
    passed_control = effect_margin > 0.0 and label_delta >= random_delta
    return {
        "dataset": bundle.name,
        "split": split_name,
        "candidate_id": test_spec.candidate_id,
        "feature_name": test_spec.feature_name,
        "feature_index": feature_index,
        "action": action,
        "control_feature_index": control_index,
        "expected_direction": test_spec.expected_direction,
        "test_effect_label_delta": label_delta,
        "test_effect_env_delta": env_signal,
        "test_effect_selectivity": selectivity,
        "test_random_control_delta": random_delta,
        "test_control_selectivity": control_selectivity,
        "test_prediction_flip_rate": flip_rate,
        "test_passed_control": passed_control,
        "test_cost": test_spec.cost,
        "label_signal_within_env": label_signal,
        "env_signal_within_label": env_signal,
        "control_label_signal_within_env": control_label_signal,
        "control_env_signal_within_label": control_env_signal,
        "reason_code": test_spec.reason_code,
    }


def execute_clue_tests(
    bundle: DatasetBundle,
    tests: Sequence[ClueTestSpec],
    *,
    packets: Sequence[Mapping[str, Any]] = (),
    model: FittedModel | None = None,
    split_name: str = "test",
) -> list[dict[str, Any]]:
    packets_by_id = {str(packet.get("candidate_id", "")): packet for packet in packets}
    return [
        execute_clue_test(
            bundle,
            test_spec,
            packet=packets_by_id.get(test_spec.candidate_id),
            model=model,
            split_name=split_name,
        )
        for test_spec in tests
    ]


def clue_rows_from_test_results(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        passed = bool(result.get("test_passed_control", False))
        label_delta = _safe_float(result.get("test_effect_label_delta"))
        env_delta = abs(_safe_float(result.get("test_effect_env_delta")))
        random_delta = abs(_safe_float(result.get("test_random_control_delta")))
        selectivity = _safe_float(result.get("test_effect_selectivity"))
        evidence = max(label_delta + selectivity - env_delta - random_delta, 0.0)
        rows.append(
            {
                "dataset": result.get("dataset", ""),
                "split": result.get("split", ""),
                "feature_name": result.get("feature_name", ""),
                "llm_hypothesis_type": "tested",
                "llm_confidence": f"{min(max(evidence + (0.25 if passed else 0.0), 0.0), 1.0):.6f}",
                "llm_reason_code": result.get("reason_code", ""),
                "llm_untested": "0",
                "test_effect_label_delta": f"{label_delta:.6f}",
                "test_effect_env_delta": f"{_safe_float(result.get('test_effect_env_delta')):.6f}",
                "test_effect_selectivity": f"{selectivity:.6f}",
                "test_random_control_delta": f"{_safe_float(result.get('test_random_control_delta')):.6f}",
                "test_prediction_flip_rate": f"{_safe_float(result.get('test_prediction_flip_rate')):.6f}",
                "test_passed_control": "1" if passed else "0",
                "test_cost": f"{_safe_float(result.get('test_cost'), default=1.0):.6f}",
            }
        )
    return rows

from __future__ import annotations

from typing import Any

import torch


def patch_tokens_from_hidden(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[1] < 2:
        raise ValueError("hidden must have shape [batch, cls_plus_patches, dim].")
    return hidden[:, 1:]


def topk_patch_mask(scores: torch.Tensor, *, top_k: int | None = None, fraction: float | None = None, largest: bool = True) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must have shape [batch, patches].")
    if top_k is None:
        if fraction is None:
            raise ValueError("Provide top_k or fraction for patch selection.")
        if fraction <= 0.0:
            raise ValueError("fraction must be positive.")
        top_k = max(1, int(round(scores.shape[1] * min(float(fraction), 1.0))))
    count = min(max(1, int(top_k)), int(scores.shape[1]))
    indices = torch.topk(scores, k=count, dim=1, largest=largest).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter(1, indices, True)


def patch_selector_scores(hidden: torch.Tensor, selector: str) -> torch.Tensor:
    cls_features = hidden[:, 0]
    patches = patch_tokens_from_hidden(hidden)
    key = selector.strip().lower().replace("-", "_")
    if key in {"cls_similarity", "cls", "attention_like"}:
        return torch.nn.functional.cosine_similarity(patches.float(), cls_features[:, None, :].float(), dim=2)
    if key in {"token_norm", "norm", "saliency"}:
        return torch.linalg.vector_norm(patches.float(), dim=2)
    raise ValueError("selector must be one of: cls_similarity, token_norm.")


def replace_patch_tokens(
    patches: torch.Tensor,
    mask: torch.Tensor,
    *,
    replacement: str,
    donor_patches: torch.Tensor | None = None,
    prototype: torch.Tensor | None = None,
) -> torch.Tensor:
    if patches.ndim != 3:
        raise ValueError("patches must have shape [batch, patches, dim].")
    if mask.shape != patches.shape[:2]:
        raise ValueError("mask must match the first two patch dimensions.")
    key = replacement.strip().lower().replace("-", "_")
    if key == "zero":
        values = torch.zeros_like(patches)
    elif key in {"mean", "batch_mean"}:
        values = patches.mean(dim=1, keepdim=True).expand_as(patches)
    elif key == "donor":
        if donor_patches is None or donor_patches.shape != patches.shape:
            raise ValueError("donor replacement requires donor_patches with the same shape as patches.")
        values = donor_patches.to(device=patches.device, dtype=patches.dtype)
    elif key in {"prototype", "centroid"}:
        if prototype is None:
            raise ValueError("prototype replacement requires a prototype tensor.")
        values = _prototype_values(prototype.to(device=patches.device, dtype=patches.dtype), patches)
    else:
        raise ValueError("replacement must be one of: zero, mean, donor, prototype.")
    return torch.where(mask.unsqueeze(-1), values, patches)


def _prototype_values(prototype: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
    if prototype.ndim == 1:
        return prototype.view(1, 1, -1).expand_as(patches)
    if prototype.ndim == 2:
        if prototype.shape[0] != patches.shape[0] or prototype.shape[1] != patches.shape[2]:
            raise ValueError("2D prototype must have shape [batch, dim].")
        return prototype[:, None, :].expand_as(patches)
    if prototype.shape != patches.shape:
        raise ValueError("3D prototype must match patches shape.")
    return prototype


def replace_hidden_patch_tokens(
    hidden: torch.Tensor,
    mask: torch.Tensor,
    *,
    replacement: str,
    donor_hidden: torch.Tensor | None = None,
    prototype: torch.Tensor | None = None,
) -> torch.Tensor:
    donor_patches = None if donor_hidden is None else patch_tokens_from_hidden(donor_hidden)
    edited_patches = replace_patch_tokens(
        patch_tokens_from_hidden(hidden),
        mask,
        replacement=replacement,
        donor_patches=donor_patches,
        prototype=prototype,
    )
    return torch.cat([hidden[:, :1], edited_patches], dim=1)


def target_logit_delta(baseline_logits: torch.Tensor, edited_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if baseline_logits.shape != edited_logits.shape:
        raise ValueError("baseline_logits and edited_logits must have the same shape.")
    labels = labels.to(dtype=torch.long, device=baseline_logits.device).view(-1)
    if labels.shape[0] != baseline_logits.shape[0]:
        raise ValueError("labels must have one entry per logit row.")
    row = torch.arange(baseline_logits.shape[0], device=baseline_logits.device)
    return edited_logits[row, labels] - baseline_logits[row, labels]


def summarize_counterfactual_effects(
    baseline_logits: torch.Tensor,
    edited_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    groups: torch.Tensor | None = None,
) -> dict[str, Any]:
    deltas = target_logit_delta(baseline_logits, edited_logits, labels).detach().cpu().float()
    baseline_pred = baseline_logits.detach().cpu().argmax(dim=1)
    edited_pred = edited_logits.detach().cpu().argmax(dim=1)
    labels_cpu = labels.detach().cpu().long().view(-1)
    summary: dict[str, Any] = {
        "mean_target_logit_delta": float(deltas.mean().item()) if deltas.numel() else 0.0,
        "mean_target_logit_drop": float((-deltas).clamp_min(0.0).mean().item()) if deltas.numel() else 0.0,
        "prediction_flip_rate": float((baseline_pred != edited_pred).float().mean().item()) if baseline_pred.numel() else 0.0,
        "correct_to_wrong_rate": float(((baseline_pred == labels_cpu) & (edited_pred != labels_cpu)).float().mean().item()) if baseline_pred.numel() else 0.0,
    }
    if groups is not None:
        groups_cpu = groups.detach().cpu().long().view(-1)
        group_effects: dict[str, float] = {}
        for group_id in torch.unique(groups_cpu):
            mask = groups_cpu == group_id
            group_effects[str(int(group_id.item()))] = float(deltas[mask].mean().item()) if bool(mask.any()) else 0.0
        summary["group_mean_target_logit_delta"] = group_effects
    return summary


def intervention_discovery_score(
    *,
    label_effect: float,
    background_effect: float = 0.0,
    random_control_effect: float = 0.0,
) -> float:
    evidence = float(label_effect) - abs(float(background_effect)) - abs(float(random_control_effect))
    return float(torch.sigmoid(torch.tensor(evidence)).item())

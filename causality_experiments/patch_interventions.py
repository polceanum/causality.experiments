from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F


class PatchFlipProbe(torch.nn.Module):
    def __init__(self, token_dim: int, *, hidden_dim: int = 0, initial_mask_probability: float | None = None) -> None:
        super().__init__()
        input_dim = int(token_dim) * 3
        if hidden_dim > 0:
            self.scorer = torch.nn.Sequential(
                torch.nn.Linear(input_dim, int(hidden_dim)),
                torch.nn.ReLU(),
                torch.nn.Linear(int(hidden_dim), 1),
            )
        else:
            self.scorer = torch.nn.Linear(input_dim, 1)
        if initial_mask_probability is not None:
            probability = min(max(float(initial_mask_probability), 1e-4), 1.0 - 1e-4)
            final = self.scorer[-1] if isinstance(self.scorer, torch.nn.Sequential) else self.scorer
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.constant_(final.bias, math.log(probability / (1.0 - probability)))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        cls_features = hidden[:, 0]
        patches = patch_tokens_from_hidden(hidden)
        cls_context = cls_features[:, None, :].expand_as(patches)
        interactions = patches * cls_context
        features = torch.cat([patches, cls_context, interactions], dim=2)
        return self.scorer(features).squeeze(-1)


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


def soft_patch_mask(mask_logits: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    if mask_logits.ndim != 2:
        raise ValueError("mask_logits must have shape [batch, patches].")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    return torch.sigmoid(mask_logits / float(temperature))


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


def replace_patch_tokens_soft(
    patches: torch.Tensor,
    mask_weights: torch.Tensor,
    *,
    replacement: str,
    donor_patches: torch.Tensor | None = None,
    prototype: torch.Tensor | None = None,
) -> torch.Tensor:
    if patches.ndim != 3:
        raise ValueError("patches must have shape [batch, patches, dim].")
    if mask_weights.shape != patches.shape[:2]:
        raise ValueError("mask_weights must match the first two patch dimensions.")
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
    weights = mask_weights.to(device=patches.device, dtype=patches.dtype).clamp(0.0, 1.0).unsqueeze(-1)
    return patches * (1.0 - weights) + values * weights


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


def replace_hidden_patch_tokens_soft(
    hidden: torch.Tensor,
    mask_weights: torch.Tensor,
    *,
    replacement: str,
    donor_hidden: torch.Tensor | None = None,
    prototype: torch.Tensor | None = None,
) -> torch.Tensor:
    donor_patches = None if donor_hidden is None else patch_tokens_from_hidden(donor_hidden)
    edited_patches = replace_patch_tokens_soft(
        patch_tokens_from_hidden(hidden),
        mask_weights,
        replacement=replacement,
        donor_patches=donor_patches,
        prototype=prototype,
    )
    return torch.cat([hidden[:, :1], edited_patches], dim=1)


def flipped_binary_targets(logits: torch.Tensor, labels: torch.Tensor | None = None, *, mode: str = "prediction") -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("flipped_binary_targets currently requires binary logits with shape [batch, 2].")
    key = mode.strip().lower().replace("-", "_")
    if key in {"prediction", "opposite_prediction", "decision"}:
        source = logits.detach().argmax(dim=1)
    elif key in {"label", "opposite_label", "target"}:
        if labels is None:
            raise ValueError("label target mode requires labels.")
        source = labels.detach().to(device=logits.device, dtype=torch.long).view(-1)
    else:
        raise ValueError("target mode must be one of: prediction, label.")
    if source.shape[0] != logits.shape[0]:
        raise ValueError("labels must have one entry per logit row.")
    return 1 - source


def counterfactual_probe_loss(
    edited_logits: torch.Tensor,
    flip_targets: torch.Tensor,
    mask_weights: torch.Tensor,
    *,
    sparsity_weight: float = 0.0,
    budget: float | None = None,
    budget_weight: float = 0.0,
    entropy_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    targets = flip_targets.to(device=edited_logits.device, dtype=torch.long).view(-1)
    if targets.shape[0] != edited_logits.shape[0]:
        raise ValueError("flip_targets must have one entry per logit row.")
    weights = mask_weights.float().clamp(1e-6, 1.0 - 1e-6)
    flip_loss = F.cross_entropy(edited_logits, targets)
    sparsity_loss = weights.mean()
    loss = flip_loss + float(sparsity_weight) * sparsity_loss
    budget_loss = weights.new_tensor(0.0)
    if budget is not None and budget_weight > 0.0:
        budget_loss = (sparsity_loss - float(budget)).pow(2)
        loss = loss + float(budget_weight) * budget_loss
    entropy_loss = -(weights * weights.log() + (1.0 - weights) * (1.0 - weights).log()).mean()
    if entropy_weight > 0.0:
        loss = loss + float(entropy_weight) * entropy_loss
    return loss, {
        "flip_loss": float(flip_loss.detach().cpu().item()),
        "sparsity_loss": float(sparsity_loss.detach().cpu().item()),
        "budget_loss": float(budget_loss.detach().cpu().item()),
        "entropy_loss": float(entropy_loss.detach().cpu().item()),
        "loss": float(loss.detach().cpu().item()),
    }


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

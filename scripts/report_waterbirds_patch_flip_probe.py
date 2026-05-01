from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.data import DatasetBundle
from causality_experiments.methods import OfficialDFRClassifier, fit_method
from causality_experiments.metrics import accuracy, worst_group_accuracy
from causality_experiments.patch_interventions import (
    PatchFlipProbe,
    counterfactual_probe_loss,
    flipped_binary_targets,
    patch_selector_scores,
    replace_hidden_patch_tokens,
    replace_hidden_patch_tokens_soft,
    soft_patch_mask,
    summarize_counterfactual_effects,
    topk_patch_mask,
)
from scripts.prepare_waterbirds_features import (
    DATASET_DIRNAME,
    DEFAULT_DOWNLOAD_DIR,
    DEFAULT_RAW_DIR,
    WaterbirdsImageDataset,
    choose_device,
    ensure_downloaded,
    ensure_extracted,
    load_metadata,
    _feature_columns_for_components,
    _hf_patch_component_features,
)


@dataclass
class HiddenBundle:
    hidden: dict[str, torch.Tensor]
    labels: dict[str, torch.Tensor]
    groups: dict[str, torch.Tensor]


class PatchFlipMixtureProbe(torch.nn.Module):
    def __init__(
        self,
        token_dim: int,
        *,
        component_count: int,
        hidden_dim: int = 0,
        initial_mask_probability: float | None = None,
    ) -> None:
        super().__init__()
        if component_count <= 0:
            raise ValueError("component_count must be positive.")
        self.component_count = int(component_count)
        self.components = torch.nn.ModuleList(
            [
                PatchFlipProbe(
                    token_dim,
                    hidden_dim=hidden_dim,
                    initial_mask_probability=initial_mask_probability,
                )
                for _ in range(self.component_count)
            ]
        )
        if hidden_dim > 0:
            self.weight_head = torch.nn.Sequential(
                torch.nn.Linear(int(token_dim), int(hidden_dim)),
                torch.nn.ReLU(),
                torch.nn.Linear(int(hidden_dim), self.component_count),
            )
        else:
            self.weight_head = torch.nn.Linear(int(token_dim), self.component_count)
        final = self.weight_head[-1] if isinstance(self.weight_head, torch.nn.Sequential) else self.weight_head
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask_logits = torch.stack([component(hidden) for component in self.components], dim=1)
        component_logits = self.weight_head(hidden[:, 0].float())
        return mask_logits, component_logits


ProbeModel = PatchFlipProbe | PatchFlipMixtureProbe


def _write_rows(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = fieldnames or sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _component_names_for_pooling(pooling: str) -> list[str]:
    key = pooling.strip().lower().replace("-", "_")
    if key in {"center_background", "patch_center_background", "patch_components"}:
        return ["cls", "center", "background", "center_minus_background"]
    if key in {"cls_similarity", "patch_cls_similarity", "patch_cls_components"}:
        return ["cls", "foreground", "background", "foreground_minus_background"]
    if key in {"token_norm", "patch_token_norm", "patch_norm_components"}:
        return ["cls", "foreground", "background", "foreground_minus_background"]
    raise ValueError("Patch component pooling must be one of: center_background, cls_similarity, token_norm.")


def component_feature_names(feature_dim: int, *, pooling: str) -> list[str]:
    columns, _ = _feature_columns_for_components(feature_dim, _component_names_for_pooling(pooling))
    return columns


def _row_minmax_unit(values: torch.Tensor) -> torch.Tensor:
    minimum = values.min(dim=1, keepdim=True).values
    maximum = values.max(dim=1, keepdim=True).values
    width = maximum - minimum
    return torch.where(width > 1e-12, (values - minimum) / width.clamp_min(1e-12), torch.full_like(values, 0.5))


def normalized_patch_prior_scores(hidden: torch.Tensor, selector: str) -> torch.Tensor:
    key = selector.strip().lower().replace("-", "_")
    if key in {"", "none", "uniform"}:
        return torch.full(hidden.shape[:2][0:1] + (hidden.shape[1] - 1,), 0.5, dtype=hidden.dtype, device=hidden.device)
    if key == "mixed":
        cls_scores = _row_minmax_unit(patch_selector_scores(hidden, "cls_similarity"))
        norm_scores = _row_minmax_unit(patch_selector_scores(hidden, "token_norm"))
        return 0.5 * (cls_scores + norm_scores)
    if key in {"cls_similarity", "token_norm"}:
        return _row_minmax_unit(patch_selector_scores(hidden, key))
    raise ValueError("Patch prior selector must be one of: none, cls_similarity, token_norm, mixed.")


def patch_prior_logit_offset(hidden: torch.Tensor, *, selector: str, budget: float) -> torch.Tensor:
    scores = normalized_patch_prior_scores(hidden, selector)
    eps = 1e-4
    probabilities = scores.clamp(eps, 1.0 - eps)
    budget_probability = min(max(float(budget), eps), 1.0 - eps)
    offset = torch.logit(probabilities) - math.log(budget_probability / (1.0 - budget_probability))
    return offset - offset.mean(dim=1, keepdim=True)


def adapted_probe_logits(
    probe: PatchFlipProbe,
    hidden: torch.Tensor,
    *,
    prior_selector: str,
    prior_weight: float,
    budget: float,
) -> torch.Tensor:
    logits = probe(hidden)
    if float(prior_weight) == 0.0 or prior_selector.strip().lower() in {"", "none", "uniform"}:
        return logits
    return logits + float(prior_weight) * patch_prior_logit_offset(hidden, selector=prior_selector, budget=budget)


def adapted_mixture_probe_outputs(
    probe: PatchFlipMixtureProbe,
    hidden: torch.Tensor,
    *,
    prior_selector: str,
    prior_weight: float,
    budget: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_logits, component_logits = probe(hidden)
    if float(prior_weight) != 0.0 and prior_selector.strip().lower() not in {"", "none", "uniform"}:
        prior_offset = patch_prior_logit_offset(hidden, selector=prior_selector, budget=budget)
        mask_logits = mask_logits + float(prior_weight) * prior_offset[:, None, :]
    return mask_logits, component_logits


def mixture_mask_scores(
    probe: PatchFlipMixtureProbe,
    hidden: torch.Tensor,
    *,
    mode: str,
    prior_selector: str,
    prior_weight: float,
    budget: float,
    component_index: int | None = None,
) -> torch.Tensor:
    mask_logits, component_logits = adapted_mixture_probe_outputs(
        probe,
        hidden,
        prior_selector=prior_selector,
        prior_weight=prior_weight,
        budget=budget,
    )
    key = mode.strip().lower().replace("-", "_")
    if key == "marginal":
        component_weights = torch.softmax(component_logits, dim=1)
        mask_weights = soft_patch_mask(mask_logits.flatten(0, 1)).view_as(mask_logits)
        return (component_weights[:, :, None] * mask_weights).sum(dim=1)
    if key == "map_component":
        selected = component_logits.argmax(dim=1)
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        return mask_logits[batch_indices, selected]
    if key == "component":
        if component_index is None:
            raise ValueError("component mode requires component_index.")
        if component_index < 0 or component_index >= probe.component_count:
            raise ValueError("component_index is out of range for the mixture probe.")
        return mask_logits[:, int(component_index)]
    raise ValueError("Mixture mask mode must be one of: marginal, map_component, component.")


def mixture_component_diversity_loss(mask_weights: torch.Tensor) -> torch.Tensor:
    if mask_weights.ndim != 3:
        raise ValueError("mask_weights must have shape [batch, components, patches].")
    component_count = int(mask_weights.shape[1])
    if component_count <= 1:
        return mask_weights.new_tensor(0.0)
    losses: list[torch.Tensor] = []
    for left_index in range(component_count):
        left = mask_weights[:, left_index]
        for right_index in range(left_index + 1, component_count):
            right = mask_weights[:, right_index]
            losses.append(torch.nn.functional.cosine_similarity(left, right, dim=1).mean())
    return torch.stack(losses).mean()


def mixture_effect_best_mask(
    *,
    probe: PatchFlipMixtureProbe,
    hidden: torch.Tensor,
    classifier: OfficialDFRClassifier,
    pooling: str,
    top_k: int,
    replacement: str,
    prior_selector: str,
    prior_weight: float,
    budget: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    baseline_features = pooled_component_features(hidden, pooling=pooling)
    baseline_logits = classifier.predict(baseline_features)
    decision_targets = baseline_logits.argmax(dim=1)
    row_indices = torch.arange(hidden.shape[0])
    baseline_target_logits = baseline_logits[row_indices, decision_targets]
    masks: list[torch.Tensor] = []
    drops: list[torch.Tensor] = []
    for component_index in range(probe.component_count):
        component_scores = mixture_mask_scores(
            probe,
            hidden.float(),
            mode="component",
            prior_selector=prior_selector,
            prior_weight=prior_weight,
            budget=budget,
            component_index=component_index,
        )
        component_mask = topk_patch_mask(component_scores, top_k=top_k, largest=True)
        edited_hidden = replace_hidden_patch_tokens(hidden.float(), component_mask, replacement=replacement)
        edited_features = pooled_component_features(edited_hidden, pooling=pooling)
        edited_logits = classifier.predict(edited_features)
        edited_target_logits = edited_logits[row_indices, decision_targets]
        masks.append(component_mask)
        drops.append(baseline_target_logits - edited_target_logits)
    mask_stack = torch.stack(masks, dim=1)
    drop_stack = torch.stack(drops, dim=1)
    best_indices = drop_stack.argmax(dim=1)
    selected_mask = mask_stack[row_indices, best_indices]
    return selected_mask, best_indices.cpu(), drop_stack.detach().cpu()


def _flatten_summary(prefix: str, summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            for group, group_value in value.items():
                row[f"{prefix}_{key}_group_{group}"] = group_value
        else:
            row[f"{prefix}_{key}"] = value
    return row


def load_hf_hidden_model(model_id: str, *, local_files_only: bool) -> tuple[torch.nn.Module, Any]:
    try:
        from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel
    except Exception as exc:
        raise RuntimeError("Patch flip probing requires transformers to be installed.") from exc

    processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    if "clip" in model_id.lower():
        model = CLIPVisionModel.from_pretrained(model_id, local_files_only=local_files_only)
    else:
        model = AutoModel.from_pretrained(model_id, local_files_only=local_files_only)
    model.eval()

    def transform(image: Any) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        encoded = processor(images=image, return_tensors="pt")
        return encoded["pixel_values"][0]

    return model, transform


def extract_hidden_bundle(
    *,
    model: torch.nn.Module,
    transform: Any,
    dataset_dir: Path,
    metadata: Any,
    device: torch.device,
    batch_size: int,
) -> HiddenBundle:
    hidden_by_split: dict[str, torch.Tensor] = {}
    labels_by_split: dict[str, torch.Tensor] = {}
    groups_by_split: dict[str, torch.Tensor] = {}
    model.eval()
    for split_name in ("train", "val", "test"):
        split_metadata = metadata[metadata["split"] == split_name].reset_index(drop=True)
        dataset = WaterbirdsImageDataset(dataset_dir, split_metadata, transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        chunks: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch, _ in loader:
                output = model(pixel_values=batch.to(device))
                hidden = getattr(output, "last_hidden_state", None)
                if hidden is None:
                    raise ValueError("Patch flip probing requires model outputs with last_hidden_state.")
                chunks.append(hidden.detach().cpu())
        hidden_by_split[split_name] = torch.cat(chunks, dim=0)
        labels_by_split[split_name] = torch.tensor(split_metadata["y"].to_numpy(), dtype=torch.long)
        groups_by_split[split_name] = torch.tensor(split_metadata["group"].to_numpy(), dtype=torch.long)
    return HiddenBundle(hidden=hidden_by_split, labels=labels_by_split, groups=groups_by_split)


def pooled_component_features(hidden: torch.Tensor, *, pooling: str) -> torch.Tensor:
    return _hf_patch_component_features(hidden, pooling=pooling).detach().cpu()


def build_component_bundle(hidden_bundle: HiddenBundle, *, pooling: str) -> DatasetBundle:
    feature_names: list[str] = []
    splits: dict[str, dict[str, torch.Tensor]] = {}
    input_dim = 0
    output_dim = 2
    for split_name in ("train", "val", "test"):
        x = pooled_component_features(hidden_bundle.hidden[split_name], pooling=pooling)
        y = hidden_bundle.labels[split_name]
        groups = hidden_bundle.groups[split_name]
        input_dim = int(x.shape[1])
        output_dim = max(output_dim, int(y.max().item()) + 1)
        if not feature_names:
            feature_names = component_feature_names(input_dim, pooling=pooling)
        splits[split_name] = {
            "x": x.float(),
            "y": y.long(),
            "env": (groups // 2).long(),
            "group": groups.long(),
        }
    return DatasetBundle(
        name="waterbirds_patch_components",
        task="classification",
        splits=splits,
        input_dim=input_dim,
        output_dim=output_dim,
        causal_mask=None,
        metadata={
            "fixture": False,
            "modality": "features",
            "patch_pooling": pooling,
            "feature_columns": feature_names,
        },
    )


def build_component_feature_rows(hidden_bundle: HiddenBundle, *, pooling: str) -> tuple[list[dict[str, Any]], list[str]]:
    first_features = pooled_component_features(hidden_bundle.hidden["train"], pooling=pooling)
    feature_names = component_feature_names(int(first_features.shape[1]), pooling=pooling)
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        features = first_features if split_name == "train" else pooled_component_features(hidden_bundle.hidden[split_name], pooling=pooling)
        labels = hidden_bundle.labels[split_name]
        groups = hidden_bundle.groups[split_name]
        places = groups // 2
        for row_index in range(int(features.shape[0])):
            row: dict[str, Any] = {
                "split": split_name,
                "y": int(labels[row_index].item()),
                "place": int(places[row_index].item()),
                "group": int(groups[row_index].item()),
            }
            row.update(
                {
                    feature_name: float(value)
                    for feature_name, value in zip(feature_names, features[row_index].tolist(), strict=True)
                }
            )
            rows.append(row)
    return rows, feature_names


def official_dfr_config(*, seed: int, retrains: int, device: str) -> dict[str, Any]:
    return {
        "name": "waterbirds_patch_flip_probe_head",
        "seed": int(seed),
        "method": {
            "kind": "official_dfr_val_tr",
            "official_dfr_c_grid": [1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01],
            "official_dfr_num_retrains": int(retrains),
            "official_dfr_balance_val": True,
            "official_dfr_add_train": False,
        },
        "training": {"device": device},
        "metrics": ["accuracy", "worst_group_accuracy"],
    }


def train_patch_flip_probe(
    *,
    hidden_train: torch.Tensor,
    classifier: OfficialDFRClassifier,
    labels: torch.Tensor,
    pooling: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    replacement: str,
    target_mode: str,
    temperature: float,
    sparsity_weight: float,
    budget: float,
    budget_weight: float,
    entropy_weight: float,
    hidden_dim: int,
    prior_selector: str,
    prior_weight: float,
) -> tuple[PatchFlipProbe, list[dict[str, float]]]:
    torch.manual_seed(seed)
    probe = PatchFlipProbe(int(hidden_train.shape[2]), hidden_dim=hidden_dim, initial_mask_probability=budget)
    optimizer = torch.optim.Adam(probe.parameters(), lr=float(lr), weight_decay=1e-4)
    baseline_logits = classifier.predict(pooled_component_features(hidden_train, pooling=pooling))
    flip_targets = flipped_binary_targets(baseline_logits, labels, mode=target_mode)
    dataset = TensorDataset(hidden_train.float(), labels.long(), flip_targets.long())
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
        totals: dict[str, float] = {}
        seen = 0
        probe.train()
        for hidden_batch, _, target_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            mask_logits = adapted_probe_logits(
                probe,
                hidden_batch,
                prior_selector=prior_selector,
                prior_weight=prior_weight,
                budget=budget,
            )
            mask_weights = soft_patch_mask(mask_logits, temperature=temperature)
            edited_hidden = replace_hidden_patch_tokens_soft(hidden_batch, mask_weights, replacement=replacement)
            edited_features = _hf_patch_component_features(edited_hidden, pooling=pooling)
            edited_logits = classifier.predict(edited_features)
            loss, parts = counterfactual_probe_loss(
                edited_logits,
                target_batch,
                mask_weights,
                sparsity_weight=sparsity_weight,
                budget=budget,
                budget_weight=budget_weight,
                entropy_weight=entropy_weight,
            )
            loss.backward()
            optimizer.step()
            batch_count = int(hidden_batch.shape[0])
            seen += batch_count
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch_count
        history.append({"epoch": float(epoch + 1), **{key: value / max(seen, 1) for key, value in totals.items()}})
    return probe, history


def _mixture_probe_loss(
    edited_logits_by_component: torch.Tensor,
    flip_targets: torch.Tensor,
    mask_weights: torch.Tensor,
    component_logits: torch.Tensor,
    *,
    baseline_logits: torch.Tensor | None = None,
    mixture_objective: str = "nll",
    mixture_effect_weight: float = 1.0,
    mixture_routing_weight: float = 0.1,
    mixture_best_of_k_temperature: float = 1.0,
    sparsity_weight: float,
    budget: float,
    budget_weight: float,
    entropy_weight: float,
    mixture_entropy_weight: float,
    mixture_diversity_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if edited_logits_by_component.ndim != 3:
        raise ValueError("edited_logits_by_component must have shape [batch, components, classes].")
    component_count = int(edited_logits_by_component.shape[1])
    targets = flip_targets.to(device=edited_logits_by_component.device, dtype=torch.long).view(-1)
    per_component_losses = torch.stack(
        [
            torch.nn.functional.cross_entropy(
                edited_logits_by_component[:, component_index],
                targets,
                reduction="none",
            )
            for component_index in range(component_count)
        ],
        dim=1,
    )
    component_weights = torch.softmax(component_logits, dim=1)
    mixture_flip_loss = -torch.logsumexp(
        component_weights.clamp_min(1e-8).log() - per_component_losses,
        dim=1,
    ).mean()
    expected_flip_loss = (component_weights.detach() * per_component_losses).sum(dim=1).mean()
    best_indices = per_component_losses.argmin(dim=1)
    selected_flip_loss = per_component_losses.gather(1, best_indices[:, None]).squeeze(1).mean()
    best_effect_drop = edited_logits_by_component.new_tensor(0.0)
    routing_loss = edited_logits_by_component.new_tensor(0.0)
    if baseline_logits is not None:
        baseline = baseline_logits.to(device=edited_logits_by_component.device, dtype=edited_logits_by_component.dtype)
        if baseline.shape != edited_logits_by_component.shape[:1] + edited_logits_by_component.shape[2:]:
            raise ValueError("baseline_logits must have shape [batch, classes].")
        decision_targets = baseline.detach().argmax(dim=1)
        row_indices = torch.arange(baseline.shape[0], device=baseline.device)
        baseline_targets = baseline[row_indices, decision_targets]
        target_index = decision_targets[:, None, None].expand(-1, component_count, 1)
        edited_targets = edited_logits_by_component.gather(2, target_index).squeeze(2)
        effect_drops = baseline_targets[:, None] - edited_targets
        best_indices = effect_drops.argmax(dim=1)
        best_effect_drop = effect_drops.gather(1, best_indices[:, None]).squeeze(1).mean()
        selected_flip_loss = per_component_losses.gather(1, best_indices[:, None]).squeeze(1).mean()
    objective_key = mixture_objective.strip().lower().replace("-", "_")
    if objective_key == "nll":
        objective_loss = mixture_flip_loss
    elif objective_key in {"best_of_k", "best", "min"}:
        temperature = max(float(mixture_best_of_k_temperature), 1e-6)
        objective_loss = (-temperature * torch.logsumexp(-per_component_losses / temperature, dim=1)).mean()
    elif objective_key in {"effect_best", "best_effect"}:
        if baseline_logits is None:
            raise ValueError("effect_best mixture objective requires baseline_logits.")
        objective_loss = selected_flip_loss - float(mixture_effect_weight) * best_effect_drop
        if mixture_routing_weight > 0.0:
            routing_loss = torch.nn.functional.cross_entropy(component_logits, best_indices.detach())
            objective_loss = objective_loss + float(mixture_routing_weight) * routing_loss
    else:
        raise ValueError("mixture_objective must be one of: nll, best_of_k, effect_best.")
    weights = mask_weights.float().clamp(1e-6, 1.0 - 1e-6)
    expected_mask = (component_weights[:, :, None] * weights).sum(dim=1)
    sparsity_loss = expected_mask.mean()
    budget_loss = (sparsity_loss - float(budget)).pow(2)
    mask_entropy_loss = -(weights * weights.log() + (1.0 - weights) * (1.0 - weights).log()).mean()
    mixture_entropy = -(component_weights.clamp_min(1e-8) * component_weights.clamp_min(1e-8).log()).sum(dim=1).mean()
    diversity_loss = mixture_component_diversity_loss(weights)
    loss = objective_loss + float(sparsity_weight) * sparsity_loss + float(budget_weight) * budget_loss
    if entropy_weight > 0.0:
        loss = loss + float(entropy_weight) * mask_entropy_loss
    if mixture_diversity_weight > 0.0:
        loss = loss + float(mixture_diversity_weight) * diversity_loss
    if mixture_entropy_weight > 0.0:
        loss = loss - float(mixture_entropy_weight) * mixture_entropy
    return loss, {
        "flip_loss": float(mixture_flip_loss.detach().cpu().item()),
        "objective_loss": float(objective_loss.detach().cpu().item()),
        "selected_flip_loss": float(selected_flip_loss.detach().cpu().item()),
        "best_effect_drop": float(best_effect_drop.detach().cpu().item()),
        "routing_loss": float(routing_loss.detach().cpu().item()),
        "expected_flip_loss": float(expected_flip_loss.detach().cpu().item()),
        "sparsity_loss": float(sparsity_loss.detach().cpu().item()),
        "budget_loss": float(budget_loss.detach().cpu().item()),
        "entropy_loss": float(mask_entropy_loss.detach().cpu().item()),
        "mixture_entropy": float(mixture_entropy.detach().cpu().item()),
        "diversity_loss": float(diversity_loss.detach().cpu().item()),
        "loss": float(loss.detach().cpu().item()),
    }


def train_patch_flip_mixture_probe(
    *,
    hidden_train: torch.Tensor,
    classifier: OfficialDFRClassifier,
    labels: torch.Tensor,
    pooling: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    replacement: str,
    target_mode: str,
    temperature: float,
    sparsity_weight: float,
    budget: float,
    budget_weight: float,
    entropy_weight: float,
    hidden_dim: int,
    prior_selector: str,
    prior_weight: float,
    component_count: int,
    mixture_objective: str,
    mixture_effect_weight: float,
    mixture_routing_weight: float,
    mixture_best_of_k_temperature: float,
    mixture_entropy_weight: float,
    mixture_diversity_weight: float,
) -> tuple[PatchFlipMixtureProbe, list[dict[str, float]]]:
    torch.manual_seed(seed)
    probe = PatchFlipMixtureProbe(
        int(hidden_train.shape[2]),
        component_count=int(component_count),
        hidden_dim=hidden_dim,
        initial_mask_probability=budget,
    )
    optimizer = torch.optim.Adam(probe.parameters(), lr=float(lr), weight_decay=1e-4)
    baseline_logits = classifier.predict(pooled_component_features(hidden_train, pooling=pooling))
    flip_targets = flipped_binary_targets(baseline_logits, labels, mode=target_mode)
    dataset = TensorDataset(hidden_train.float(), labels.long(), flip_targets.long(), baseline_logits.float())
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
        totals: dict[str, float] = {}
        seen = 0
        probe.train()
        for hidden_batch, _, target_batch, baseline_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            mask_logits, component_logits = adapted_mixture_probe_outputs(
                probe,
                hidden_batch,
                prior_selector=prior_selector,
                prior_weight=prior_weight,
                budget=budget,
            )
            mask_weights = soft_patch_mask(mask_logits.flatten(0, 1), temperature=temperature).view_as(mask_logits)
            edited_logits: list[torch.Tensor] = []
            for component_index in range(int(component_count)):
                edited_hidden = replace_hidden_patch_tokens_soft(
                    hidden_batch,
                    mask_weights[:, component_index],
                    replacement=replacement,
                )
                edited_features = _hf_patch_component_features(edited_hidden, pooling=pooling)
                edited_logits.append(classifier.predict(edited_features))
            edited_logits_by_component = torch.stack(edited_logits, dim=1)
            loss, parts = _mixture_probe_loss(
                edited_logits_by_component,
                target_batch,
                mask_weights,
                component_logits,
                baseline_logits=baseline_batch,
                mixture_objective=mixture_objective,
                mixture_effect_weight=mixture_effect_weight,
                mixture_routing_weight=mixture_routing_weight,
                mixture_best_of_k_temperature=mixture_best_of_k_temperature,
                sparsity_weight=sparsity_weight,
                budget=budget,
                budget_weight=budget_weight,
                entropy_weight=entropy_weight,
                mixture_entropy_weight=mixture_entropy_weight,
                mixture_diversity_weight=mixture_diversity_weight,
            )
            loss.backward()
            optimizer.step()
            batch_count = int(hidden_batch.shape[0])
            seen += batch_count
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch_count
        history.append({"epoch": float(epoch + 1), **{key: value / max(seen, 1) for key, value in totals.items()}})
    return probe, history


def _hard_mask_from_strategy(
    *,
    strategy: str,
    hidden: torch.Tensor,
    top_k: int,
    probe: ProbeModel | None,
    seed: int,
    prior_selector: str = "none",
    prior_weight: float = 0.0,
    budget: float = 0.10,
    mixture_component_index: int | None = None,
) -> torch.Tensor:
    key = strategy.strip().lower()
    if key == "learned_probe":
        if probe is None:
            raise ValueError("learned_probe strategy requires a trained probe.")
        if isinstance(probe, PatchFlipMixtureProbe):
            with torch.inference_mode():
                scores = mixture_mask_scores(
                    probe,
                    hidden.float(),
                    mode="marginal",
                    prior_selector=prior_selector,
                    prior_weight=prior_weight,
                    budget=budget,
                )
            return topk_patch_mask(scores, top_k=top_k, largest=True)
        probe.eval()
        with torch.inference_mode():
            scores = adapted_probe_logits(
                probe,
                hidden.float(),
                prior_selector=prior_selector,
                prior_weight=prior_weight,
                budget=budget,
            )
        return topk_patch_mask(scores, top_k=top_k, largest=True)
    if key in {"mixture_marginal", "mixture_posterior_marginal"}:
        if not isinstance(probe, PatchFlipMixtureProbe):
            raise ValueError("mixture_marginal strategy requires a mixture probe.")
        probe.eval()
        with torch.inference_mode():
            scores = mixture_mask_scores(
                probe,
                hidden.float(),
                mode="marginal",
                prior_selector=prior_selector,
                prior_weight=prior_weight,
                budget=budget,
            )
        return topk_patch_mask(scores, top_k=top_k, largest=True)
    if key in {"mixture_map_component", "mixture_map"}:
        if not isinstance(probe, PatchFlipMixtureProbe):
            raise ValueError("mixture_map_component strategy requires a mixture probe.")
        probe.eval()
        with torch.inference_mode():
            scores = mixture_mask_scores(
                probe,
                hidden.float(),
                mode="map_component",
                prior_selector=prior_selector,
                prior_weight=prior_weight,
                budget=budget,
            )
        return topk_patch_mask(scores, top_k=top_k, largest=True)
    if key in {"mixture_val_best_component", "mixture_component"}:
        if not isinstance(probe, PatchFlipMixtureProbe):
            raise ValueError("mixture component strategies require a mixture probe.")
        probe.eval()
        with torch.inference_mode():
            scores = mixture_mask_scores(
                probe,
                hidden.float(),
                mode="component",
                prior_selector=prior_selector,
                prior_weight=prior_weight,
                budget=budget,
                component_index=mixture_component_index,
            )
        return topk_patch_mask(scores, top_k=top_k, largest=True)
    if key == "cls_similarity_top":
        return topk_patch_mask(patch_selector_scores(hidden, "cls_similarity"), top_k=top_k, largest=True)
    if key == "cls_similarity_bottom":
        return topk_patch_mask(patch_selector_scores(hidden, "cls_similarity"), top_k=top_k, largest=False)
    if key == "token_norm_top":
        return topk_patch_mask(patch_selector_scores(hidden, "token_norm"), top_k=top_k, largest=True)
    if key == "random":
        generator = torch.Generator().manual_seed(seed)
        return topk_patch_mask(torch.rand(hidden.shape[0], hidden.shape[1] - 1, generator=generator), top_k=top_k)
    raise ValueError(f"Unknown intervention strategy {strategy!r}.")


def evaluate_intervention_strategy(
    *,
    strategy: str,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    classifier: OfficialDFRClassifier,
    pooling: str,
    top_k: int,
    replacement: str,
    probe: ProbeModel | None = None,
    seed: int = 0,
    prior_selector: str = "none",
    prior_weight: float = 0.0,
    budget: float = 0.10,
    mixture_component_index: int | None = None,
) -> dict[str, Any]:
    baseline_features = pooled_component_features(hidden, pooling=pooling)
    baseline_logits = classifier.predict(baseline_features)
    selected_components: torch.Tensor | None = None
    if strategy.strip().lower().replace("-", "_") == "mixture_effect_best_component":
        if not isinstance(probe, PatchFlipMixtureProbe):
            raise ValueError("mixture_effect_best_component strategy requires a mixture probe.")
        mask, selected_components, _ = mixture_effect_best_mask(
            probe=probe,
            hidden=hidden,
            classifier=classifier,
            pooling=pooling,
            top_k=top_k,
            replacement=replacement,
            prior_selector=prior_selector,
            prior_weight=prior_weight,
            budget=budget,
        )
    else:
        mask = _hard_mask_from_strategy(
            strategy=strategy,
            hidden=hidden,
            top_k=top_k,
            probe=probe,
            seed=seed,
            prior_selector=prior_selector,
            prior_weight=prior_weight,
            budget=budget,
            mixture_component_index=mixture_component_index,
        )
    edited_hidden = replace_hidden_patch_tokens(hidden.float(), mask, replacement=replacement)
    edited_features = pooled_component_features(edited_hidden, pooling=pooling)
    edited_logits = classifier.predict(edited_features)
    label_summary = summarize_counterfactual_effects(baseline_logits, edited_logits, labels, groups=groups)
    decision_summary = summarize_counterfactual_effects(
        baseline_logits,
        edited_logits,
        baseline_logits.argmax(dim=1),
        groups=groups,
    )
    row: dict[str, Any] = {
        "strategy": strategy,
        "top_k": int(top_k),
        "replacement": replacement,
        "mean_mask_fraction": float(mask.float().mean().item()),
        "baseline_accuracy": float((baseline_logits.argmax(dim=1) == labels).float().mean().item()),
        "edited_accuracy": float((edited_logits.argmax(dim=1) == labels).float().mean().item()),
    }
    if mixture_component_index is not None:
        row["mixture_component_index"] = int(mixture_component_index)
    if selected_components is not None:
        row["mean_mixture_component_index"] = float(selected_components.float().mean().item())
    row.update(_flatten_summary("label", label_summary))
    row.update(_flatten_summary("decision", decision_summary))
    return row


def select_mixture_component_by_validation(
    *,
    probe: PatchFlipMixtureProbe,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    classifier: OfficialDFRClassifier,
    pooling: str,
    top_k: int,
    replacement: str,
    prior_selector: str,
    prior_weight: float,
    budget: float,
) -> tuple[int, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for component_index in range(probe.component_count):
        row = evaluate_intervention_strategy(
            strategy="mixture_component",
            hidden=hidden,
            labels=labels,
            groups=groups,
            classifier=classifier,
            pooling=pooling,
            top_k=top_k,
            replacement=replacement,
            probe=probe,
            prior_selector=prior_selector,
            prior_weight=prior_weight,
            budget=budget,
            mixture_component_index=component_index,
        )
        row["strategy"] = f"mixture_component_{component_index}"
        rows.append(row)
    best_row = max(rows, key=lambda row: float(row.get("decision_mean_target_logit_drop", 0.0)))
    return int(best_row["mixture_component_index"]), rows


def _target_feature_weights(classifier: OfficialDFRClassifier, targets: torch.Tensor) -> torch.Tensor:
    weights = classifier.weight.float()
    targets = targets.long()
    if classifier.output_dim == 2 and weights.shape[0] == 1:
        signs = torch.where(targets == 1, 1.0, -1.0).to(dtype=weights.dtype)
        return signs[:, None] * weights[0][None, :]
    return weights[targets]


def _normalise_positive_scores(values: torch.Tensor) -> torch.Tensor:
    values = values.float().clamp_min(0.0)
    maximum = values.max().clamp_min(1e-12)
    if float(maximum.item()) <= 1e-12:
        return torch.zeros_like(values)
    return values / maximum


def build_intervention_feature_score_rows(
    *,
    strategy: str,
    hidden: torch.Tensor,
    classifier: OfficialDFRClassifier,
    pooling: str,
    top_k: int,
    replacement: str,
    probe: ProbeModel | None = None,
    seed: int = 0,
    dataset_name: str = "waterbirds_patch_components",
    prior_selector: str = "none",
    prior_weight: float = 0.0,
    budget: float = 0.10,
    mixture_component_index: int | None = None,
) -> list[dict[str, Any]]:
    baseline_features = pooled_component_features(hidden, pooling=pooling)
    baseline_logits = classifier.predict(baseline_features)
    decision_targets = baseline_logits.argmax(dim=1)
    if strategy.strip().lower().replace("-", "_") == "mixture_effect_best_component":
        if not isinstance(probe, PatchFlipMixtureProbe):
            raise ValueError("mixture_effect_best_component strategy requires a mixture probe.")
        mask, _, _ = mixture_effect_best_mask(
            probe=probe,
            hidden=hidden,
            classifier=classifier,
            pooling=pooling,
            top_k=top_k,
            replacement=replacement,
            prior_selector=prior_selector,
            prior_weight=prior_weight,
            budget=budget,
        )
    else:
        mask = _hard_mask_from_strategy(
            strategy=strategy,
            hidden=hidden,
            top_k=top_k,
            probe=probe,
            seed=seed,
            prior_selector=prior_selector,
            prior_weight=prior_weight,
            budget=budget,
            mixture_component_index=mixture_component_index,
        )
    edited_hidden = replace_hidden_patch_tokens(hidden.float(), mask, replacement=replacement)
    edited_features = pooled_component_features(edited_hidden, pooling=pooling)
    target_weights = _target_feature_weights(classifier, decision_targets)
    contributions = (baseline_features - edited_features) * target_weights
    positive_drop = contributions.clamp_min(0.0).mean(dim=0)
    mean_drop = contributions.mean(dim=0)
    mean_abs_delta = contributions.abs().mean(dim=0)
    scores = _normalise_positive_scores(positive_drop)
    feature_names = component_feature_names(int(baseline_features.shape[1]), pooling=pooling)
    return [
        {
            "dataset": dataset_name,
            "feature_index": feature_index,
            "feature_name": feature_name,
            "score": f"{float(scores[feature_index].item()):.6f}",
            "raw_score": f"{float(positive_drop[feature_index].item()):.9f}",
            "mean_target_logit_drop": f"{float(mean_drop[feature_index].item()):.9f}",
            "mean_abs_target_logit_delta": f"{float(mean_abs_delta[feature_index].item()):.9f}",
            "strategy": strategy,
            "replacement": replacement,
            "top_k": int(top_k),
        }
        for feature_index, feature_name in enumerate(feature_names)
    ]


def build_excess_feature_score_rows(
    primary_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    *,
    strategy: str,
) -> list[dict[str, Any]]:
    control_by_name = {str(row["feature_name"]): row for row in control_rows}
    excess_values: list[float] = []
    for row in primary_rows:
        control = control_by_name.get(str(row["feature_name"]))
        control_score = 0.0 if control is None else float(control.get("raw_score", 0.0))
        excess_values.append(max(float(row.get("raw_score", 0.0)) - control_score, 0.0))
    normalised = _normalise_positive_scores(torch.tensor(excess_values, dtype=torch.float32))
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(primary_rows):
        output = dict(row)
        output["score"] = f"{float(normalised[index].item()):.6f}"
        output["raw_score"] = f"{excess_values[index]:.9f}"
        output["strategy"] = strategy
        rows.append(output)
    return rows


def compact_official_details(classifier: OfficialDFRClassifier) -> dict[str, Any]:
    details = classifier.details or {}
    keys = (
        "official_dfr_best_c",
        "official_dfr_best_feature_scale",
        "official_dfr_best_tune_wga",
        "official_dfr_balance_val",
        "official_dfr_add_train",
        "official_dfr_num_retrains",
        "official_dfr_tune_val_group_counts",
    )
    return {key: details[key] for key in keys if key in details}


def _float_rows(rows: list[dict[str, float]]) -> list[dict[str, Any]]:
    return [{key: float(value) for key, value in row.items()} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--download-dir", default=str(DEFAULT_DOWNLOAD_DIR))
    parser.add_argument("--model-id", default="facebook/dinov2-small")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=384)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pooling", default="cls_similarity", choices=("center_background", "cls_similarity", "token_norm"))
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--official-dfr-num-retrains", type=int, default=10)
    parser.add_argument("--train-epochs", type=int, default=25)
    parser.add_argument("--probe-hidden-dim", type=int, default=128)
    parser.add_argument("--probe-components", type=int, default=1)
    parser.add_argument("--probe-lr", type=float, default=0.01)
    parser.add_argument("--mask-fraction", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sparsity-weight", type=float, default=0.0)
    parser.add_argument("--budget-weight", type=float, default=20.0)
    parser.add_argument("--entropy-weight", type=float, default=0.001)
    parser.add_argument("--mixture-objective", default="nll", choices=("nll", "best_of_k", "effect_best"))
    parser.add_argument("--mixture-effect-weight", type=float, default=1.0)
    parser.add_argument("--mixture-routing-weight", type=float, default=0.1)
    parser.add_argument("--mixture-best-of-k-temperature", type=float, default=1.0)
    parser.add_argument("--mixture-entropy-weight", type=float, default=0.01)
    parser.add_argument("--mixture-diversity-weight", type=float, default=0.05)
    parser.add_argument("--probe-prior-selector", default="none", choices=("none", "cls_similarity", "token_norm", "mixed"))
    parser.add_argument("--probe-prior-weight", type=float, default=0.0)
    parser.add_argument("--replacement", default="mean", choices=("zero", "mean"))
    parser.add_argument("--target-mode", default="prediction", choices=("prediction", "label"))
    parser.add_argument("--out-dir", default="outputs/dfr_sweeps/patch_flip_probe_limit384")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    dataset_dir = raw_dir / DATASET_DIRNAME
    if not dataset_dir.joinpath("metadata.csv").exists():
        archive_path = ensure_downloaded(Path(args.download_dir), force=False)
        dataset_dir = ensure_extracted(archive_path, raw_dir, force=False)
    metadata = load_metadata(dataset_dir, limit=args.limit)
    device = choose_device(args.device)
    hf_model, transform = load_hf_hidden_model(args.model_id, local_files_only=bool(args.local_files_only))
    hf_model.to(device)
    hidden_bundle = extract_hidden_bundle(
        model=hf_model,
        transform=transform,
        dataset_dir=dataset_dir,
        metadata=metadata,
        device=device,
        batch_size=int(args.batch_size),
    )
    component_bundle = build_component_bundle(hidden_bundle, pooling=args.pooling)
    config = official_dfr_config(seed=args.seed, retrains=args.official_dfr_num_retrains, device="cpu")
    classifier = fit_method(component_bundle, config)
    if not isinstance(classifier, OfficialDFRClassifier):
        raise TypeError("Expected official DFR to return OfficialDFRClassifier.")
    baseline = {
        "val_accuracy": accuracy(classifier, component_bundle.split("val")),
        "val_wga": worst_group_accuracy(classifier, component_bundle.split("val")),
        "test_accuracy": accuracy(classifier, component_bundle.split("test")),
        "test_wga": worst_group_accuracy(classifier, component_bundle.split("test")),
        **compact_official_details(classifier),
    }
    top_k = max(1, int(round((hidden_bundle.hidden["train"].shape[1] - 1) * float(args.mask_fraction))))
    probe: ProbeModel
    if int(args.probe_components) > 1:
        probe, history = train_patch_flip_mixture_probe(
            hidden_train=hidden_bundle.hidden["train"],
            classifier=classifier,
            labels=hidden_bundle.labels["train"],
            pooling=args.pooling,
            epochs=args.train_epochs,
            batch_size=args.batch_size,
            lr=args.probe_lr,
            seed=args.seed,
            replacement=args.replacement,
            target_mode=args.target_mode,
            temperature=args.temperature,
            sparsity_weight=args.sparsity_weight,
            budget=float(args.mask_fraction),
            budget_weight=args.budget_weight,
            entropy_weight=args.entropy_weight,
            hidden_dim=args.probe_hidden_dim,
            prior_selector=args.probe_prior_selector,
            prior_weight=args.probe_prior_weight,
            component_count=int(args.probe_components),
            mixture_objective=args.mixture_objective,
            mixture_effect_weight=args.mixture_effect_weight,
            mixture_routing_weight=args.mixture_routing_weight,
            mixture_best_of_k_temperature=args.mixture_best_of_k_temperature,
            mixture_entropy_weight=args.mixture_entropy_weight,
            mixture_diversity_weight=args.mixture_diversity_weight,
        )
    else:
        probe, history = train_patch_flip_probe(
            hidden_train=hidden_bundle.hidden["train"],
            classifier=classifier,
            labels=hidden_bundle.labels["train"],
            pooling=args.pooling,
            epochs=args.train_epochs,
            batch_size=args.batch_size,
            lr=args.probe_lr,
            seed=args.seed,
            replacement=args.replacement,
            target_mode=args.target_mode,
            temperature=args.temperature,
            sparsity_weight=args.sparsity_weight,
            budget=float(args.mask_fraction),
            budget_weight=args.budget_weight,
            entropy_weight=args.entropy_weight,
            hidden_dim=args.probe_hidden_dim,
            prior_selector=args.probe_prior_selector,
            prior_weight=args.probe_prior_weight,
        )
    validation_component_rows: list[dict[str, Any]] = []
    mixture_best_component_index: int | None = None
    if isinstance(probe, PatchFlipMixtureProbe):
        mixture_best_component_index, validation_component_rows = select_mixture_component_by_validation(
            probe=probe,
            hidden=hidden_bundle.hidden["val"],
            labels=hidden_bundle.labels["val"],
            groups=hidden_bundle.groups["val"],
            classifier=classifier,
            pooling=args.pooling,
            top_k=top_k,
            replacement=args.replacement,
            prior_selector=args.probe_prior_selector,
            prior_weight=args.probe_prior_weight,
            budget=float(args.mask_fraction),
        )
    learned_strategies = (
        ["mixture_marginal", "mixture_map_component", "mixture_val_best_component", "mixture_effect_best_component"]
        if isinstance(probe, PatchFlipMixtureProbe)
        else ["learned_probe"]
    )
    control_strategies = ["cls_similarity_top", "cls_similarity_bottom", "token_norm_top", "random"]
    strategy_component_index = {
        "mixture_val_best_component": mixture_best_component_index,
    }
    rows = [
        evaluate_intervention_strategy(
            strategy=strategy,
            hidden=hidden_bundle.hidden["test"],
            labels=hidden_bundle.labels["test"],
            groups=hidden_bundle.groups["test"],
            classifier=classifier,
            pooling=args.pooling,
            top_k=top_k,
            replacement=args.replacement,
            probe=probe,
            seed=args.seed + offset,
            prior_selector=args.probe_prior_selector,
            prior_weight=args.probe_prior_weight,
            budget=float(args.mask_fraction),
            mixture_component_index=strategy_component_index.get(strategy),
        )
        for offset, strategy in enumerate([*learned_strategies, *control_strategies])
    ]
    score_rows_by_strategy = {
        strategy: build_intervention_feature_score_rows(
            strategy=strategy,
            hidden=hidden_bundle.hidden["test"],
            classifier=classifier,
            pooling=args.pooling,
            top_k=top_k,
            replacement=args.replacement,
            probe=probe,
            seed=args.seed + offset,
            prior_selector=args.probe_prior_selector,
            prior_weight=args.probe_prior_weight,
            budget=float(args.mask_fraction),
            mixture_component_index=strategy_component_index.get(strategy),
        )
        for offset, strategy in enumerate([*learned_strategies, *control_strategies])
    }
    primary_score_key = learned_strategies[0]
    score_rows_by_strategy[f"{primary_score_key}_excess_random"] = build_excess_feature_score_rows(
        score_rows_by_strategy[primary_score_key],
        score_rows_by_strategy["random"],
        strategy=f"{primary_score_key}_excess_random",
    )
    component_rows, feature_names = build_component_feature_rows(hidden_bundle, pooling=args.pooling)
    out_dir = Path(args.out_dir)
    _write_rows(out_dir / "intervention_rows.csv", rows)
    if validation_component_rows:
        _write_rows(out_dir / "validation_component_rows.csv", validation_component_rows)
    _write_rows(out_dir / "training_history.csv", _float_rows(history))
    _write_rows(out_dir / "component_features.csv", component_rows, fieldnames=["split", "y", "place", "group", *feature_names])
    score_paths: dict[str, str] = {}
    for strategy, score_rows in score_rows_by_strategy.items():
        score_path = out_dir / f"feature_scores_{strategy}.csv"
        _write_rows(score_path, score_rows)
        score_paths[strategy] = str(score_path)
    manifest = {
        "baseline": baseline,
        "component_features": str(out_dir / "component_features.csv"),
        "feature_score_paths": score_paths,
        "intervention_rows": str(out_dir / "intervention_rows.csv"),
        "validation_component_rows": str(out_dir / "validation_component_rows.csv") if validation_component_rows else "",
        "mixture_best_component_index": mixture_best_component_index,
        "training_history": str(out_dir / "training_history.csv"),
        "args": vars(args),
        "top_k": top_k,
        "patch_count": int(hidden_bundle.hidden["train"].shape[1] - 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(out_dir / "manifest.json"), "baseline": baseline}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
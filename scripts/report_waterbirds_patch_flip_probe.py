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


def _hard_mask_from_strategy(
    *,
    strategy: str,
    hidden: torch.Tensor,
    top_k: int,
    probe: PatchFlipProbe | None,
    seed: int,
    prior_selector: str = "none",
    prior_weight: float = 0.0,
    budget: float = 0.10,
) -> torch.Tensor:
    key = strategy.strip().lower()
    if key == "learned_probe":
        if probe is None:
            raise ValueError("learned_probe strategy requires a trained probe.")
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
    probe: PatchFlipProbe | None = None,
    seed: int = 0,
    prior_selector: str = "none",
    prior_weight: float = 0.0,
    budget: float = 0.10,
) -> dict[str, Any]:
    baseline_features = pooled_component_features(hidden, pooling=pooling)
    baseline_logits = classifier.predict(baseline_features)
    mask = _hard_mask_from_strategy(
        strategy=strategy,
        hidden=hidden,
        top_k=top_k,
        probe=probe,
        seed=seed,
        prior_selector=prior_selector,
        prior_weight=prior_weight,
        budget=budget,
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
    row.update(_flatten_summary("label", label_summary))
    row.update(_flatten_summary("decision", decision_summary))
    return row


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
    probe: PatchFlipProbe | None = None,
    seed: int = 0,
    dataset_name: str = "waterbirds_patch_components",
    prior_selector: str = "none",
    prior_weight: float = 0.0,
    budget: float = 0.10,
) -> list[dict[str, Any]]:
    baseline_features = pooled_component_features(hidden, pooling=pooling)
    baseline_logits = classifier.predict(baseline_features)
    decision_targets = baseline_logits.argmax(dim=1)
    mask = _hard_mask_from_strategy(
        strategy=strategy,
        hidden=hidden,
        top_k=top_k,
        probe=probe,
        seed=seed,
        prior_selector=prior_selector,
        prior_weight=prior_weight,
        budget=budget,
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
    parser.add_argument("--probe-lr", type=float, default=0.01)
    parser.add_argument("--mask-fraction", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sparsity-weight", type=float, default=0.0)
    parser.add_argument("--budget-weight", type=float, default=20.0)
    parser.add_argument("--entropy-weight", type=float, default=0.001)
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
        )
        for offset, strategy in enumerate(
            ["learned_probe", "cls_similarity_top", "cls_similarity_bottom", "token_norm_top", "random"]
        )
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
        )
        for offset, strategy in enumerate(
            ["learned_probe", "cls_similarity_top", "cls_similarity_bottom", "token_norm_top", "random"]
        )
    }
    score_rows_by_strategy["learned_probe_excess_random"] = build_excess_feature_score_rows(
        score_rows_by_strategy["learned_probe"],
        score_rows_by_strategy["random"],
        strategy="learned_probe_excess_random",
    )
    component_rows, feature_names = build_component_feature_rows(hidden_bundle, pooling=args.pooling)
    out_dir = Path(args.out_dir)
    _write_rows(out_dir / "intervention_rows.csv", rows)
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
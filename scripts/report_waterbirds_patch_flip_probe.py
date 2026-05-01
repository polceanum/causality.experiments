from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
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
    _hf_patch_component_features,
)


@dataclass
class HiddenBundle:
    hidden: dict[str, torch.Tensor]
    labels: dict[str, torch.Tensor]
    groups: dict[str, torch.Tensor]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


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
    splits: dict[str, dict[str, torch.Tensor]] = {}
    input_dim = 0
    output_dim = 2
    for split_name in ("train", "val", "test"):
        x = pooled_component_features(hidden_bundle.hidden[split_name], pooling=pooling)
        y = hidden_bundle.labels[split_name]
        groups = hidden_bundle.groups[split_name]
        input_dim = int(x.shape[1])
        output_dim = max(output_dim, int(y.max().item()) + 1)
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
        metadata={"fixture": False, "modality": "features", "patch_pooling": pooling},
    )


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
            mask_weights = soft_patch_mask(probe(hidden_batch), temperature=temperature)
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
) -> torch.Tensor:
    key = strategy.strip().lower()
    if key == "learned_probe":
        if probe is None:
            raise ValueError("learned_probe strategy requires a trained probe.")
        probe.eval()
        with torch.inference_mode():
            scores = probe(hidden.float())
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
) -> dict[str, Any]:
    baseline_features = pooled_component_features(hidden, pooling=pooling)
    baseline_logits = classifier.predict(baseline_features)
    mask = _hard_mask_from_strategy(strategy=strategy, hidden=hidden, top_k=top_k, probe=probe, seed=seed)
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
        )
        for offset, strategy in enumerate(
            ["learned_probe", "cls_similarity_top", "cls_similarity_bottom", "token_norm_top", "random"]
        )
    ]
    out_dir = Path(args.out_dir)
    _write_rows(out_dir / "intervention_rows.csv", rows)
    _write_rows(out_dir / "training_history.csv", _float_rows(history))
    manifest = {
        "baseline": baseline,
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
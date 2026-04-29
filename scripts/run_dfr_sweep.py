from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import itertools
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.config import load_config
from causality_experiments.data import load_dataset
from causality_experiments.methods import fit_method
from causality_experiments.metrics import evaluate


def _float_values(values: list[str] | None, default: float) -> list[float]:
    if not values:
        return [default]
    return [float(value) for value in values]


def _int_values(values: list[str] | None, default: int) -> list[int]:
    if not values:
        return [default]
    return [int(value) for value in values]


def _str_values(values: list[str] | None, default: str) -> list[str]:
    if not values:
        return [default]
    return values


def _row(config: dict[str, Any], metrics: dict[str, float]) -> dict[str, str]:
    method = config.get("method", {})
    dataset = config.get("dataset", {})
    return {
        "config": str(config.get("name", "")),
        "seed": str(config.get("seed", "")),
        "method": str(method.get("kind", "")),
        "dataset_path": str(dataset.get("path", "")),
        "causal_mask_top_k": str(dataset.get("causal_mask_top_k", "")),
        "causal_mask_min_margin": str(dataset.get("causal_mask_min_margin", "")),
        "dfr_optimizer": str(method.get("dfr_optimizer", "adam")),
        "dfr_split": str(method.get("dfr_split", "val")),
        "dfr_epochs": str(method.get("dfr_epochs", "")),
        "dfr_batch_size": str(method.get("dfr_batch_size", "")),
        "dfr_lr": str(method.get("dfr_lr", "")),
        "dfr_weight_decay": str(method.get("dfr_weight_decay", "")),
        "dfr_group_weight_mode": str(method.get("dfr_group_weight_mode", "sampler")),
        "dfr_group_weight_power": str(method.get("dfr_group_weight_power", "1.0")),
        "causal_dfr_nuisance_weight": str(method.get("causal_dfr_nuisance_weight", "")),
        "causal_dfr_nuisance_prior": str(method.get("causal_dfr_nuisance_prior", "mask")),
        "dfr_counterfactual_consistency_weight": str(method.get("dfr_counterfactual_consistency_weight", "0.0")),
        "validation_usage": "trains_on_validation_groups",
        "val_wga": str(metrics.get("val/worst_group_accuracy", "")),
        "test_wga": str(metrics.get("test/worst_group_accuracy", "")),
        "val_acc": str(metrics.get("val/accuracy", "")),
        "test_acc": str(metrics.get("test/accuracy", "")),
        "causal_probe": str(metrics.get("probe/causal_accuracy", "")),
        "nuisance_probe": str(metrics.get("probe/nuisance_accuracy", "")),
        "selectivity": str(metrics.get("probe/selectivity", "")),
        "nuisance_to_causal_importance": str(metrics.get("feature_importance/nuisance_to_causal", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True, help="DFR or causal_dfr config to sweep.")
    parser.add_argument("--output", default="", help="CSV output path. Defaults to outputs/dfr_sweeps/<timestamp>.csv.")
    parser.add_argument("--dataset-path", default="", help="Override dataset.path in the base config.")
    parser.add_argument("--seeds", nargs="*")
    parser.add_argument("--optimizers", nargs="*")
    parser.add_argument("--dfr-splits", nargs="*")
    parser.add_argument("--dfr-epochs", nargs="*")
    parser.add_argument("--dfr-batch-sizes", nargs="*")
    parser.add_argument("--lrs", nargs="*")
    parser.add_argument("--weight-decays", nargs="*")
    parser.add_argument("--group-weight-modes", nargs="*")
    parser.add_argument("--group-weight-powers", nargs="*")
    parser.add_argument("--causal-mask-top-ks", nargs="*")
    parser.add_argument("--causal-mask-min-margins", nargs="*")
    parser.add_argument("--nuisance-weights", nargs="*")
    parser.add_argument("--nuisance-priors", nargs="*")
    parser.add_argument("--consistency-weights", nargs="*")
    args = parser.parse_args()

    base = load_config(args.base_config)
    base_method = dict(base.get("method", {}))
    method_kind = str(base_method.get("kind", ""))
    if method_kind not in {"dfr", "causal_dfr"}:
        raise ValueError("run_dfr_sweep.py requires a base config with method.kind dfr or causal_dfr.")
    if args.dataset_path:
        base.setdefault("dataset", {})["path"] = args.dataset_path

    seeds = _int_values(args.seeds, int(base.get("seed", 0)))
    optimizers = _str_values(args.optimizers, str(base_method.get("dfr_optimizer", "adam")))
    dfr_splits = _str_values(args.dfr_splits, str(base_method.get("dfr_split", "val")))
    dfr_epochs = _int_values(args.dfr_epochs, int(base_method.get("dfr_epochs", base.get("training", {}).get("epochs", 100))))
    dfr_batch_sizes = _int_values(
        args.dfr_batch_sizes,
        int(base_method.get("dfr_batch_size", base.get("training", {}).get("batch_size", 64))),
    )
    lrs = _float_values(args.lrs, float(base_method.get("dfr_lr", base.get("training", {}).get("lr", 1e-3))))
    weight_decays = _float_values(
        args.weight_decays,
        float(base_method.get("dfr_weight_decay", base.get("training", {}).get("weight_decay", 0.0))),
    )
    group_modes = _str_values(args.group_weight_modes, str(base_method.get("dfr_group_weight_mode", "sampler")))
    group_powers = _float_values(args.group_weight_powers, float(base_method.get("dfr_group_weight_power", 1.0)))
    dataset_base = dict(base.get("dataset", {}))
    causal_mask_top_ks = _int_values(args.causal_mask_top_ks, int(dataset_base.get("causal_mask_top_k", 0) or 0))
    causal_mask_min_margins = _float_values(
        args.causal_mask_min_margins,
        float(dataset_base.get("causal_mask_min_margin", 0.0)),
    )
    nuisance_weights = _float_values(args.nuisance_weights, float(base_method.get("causal_dfr_nuisance_weight", 0.0)))
    nuisance_priors = _str_values(args.nuisance_priors, str(base_method.get("causal_dfr_nuisance_prior", "mask")))
    consistency_weights = _float_values(args.consistency_weights, float(base_method.get("dfr_counterfactual_consistency_weight", 0.0)))

    output = Path(args.output) if args.output else Path("outputs/dfr_sweeps") / f"dfr-sweep-{datetime.now():%Y%m%d-%H%M%S}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] | None = None
    bundle_cache: dict[tuple[str, int, float], Any] = {}
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer: csv.DictWriter[str] | None = None
        for seed, optimizer, dfr_split, epochs, batch_size, lr, weight_decay, group_mode, group_power, mask_top_k, mask_min_margin, nuisance_weight, nuisance_prior, consistency_weight in itertools.product(
            seeds,
            optimizers,
            dfr_splits,
            dfr_epochs,
            dfr_batch_sizes,
            lrs,
            weight_decays,
            group_modes,
            group_powers,
            causal_mask_top_ks,
            causal_mask_min_margins,
            nuisance_weights,
            nuisance_priors,
            consistency_weights,
        ):
            config = copy.deepcopy(base)
            method = dict(config.get("method", {}))
            dataset = dict(config.get("dataset", {}))
            config["seed"] = seed
            method["dfr_optimizer"] = optimizer
            method["dfr_split"] = dfr_split
            method["dfr_epochs"] = epochs
            method["dfr_batch_size"] = batch_size
            method["dfr_lr"] = lr
            method["dfr_weight_decay"] = weight_decay
            method["dfr_group_weight_mode"] = group_mode
            method["dfr_group_weight_power"] = group_power
            method["dfr_counterfactual_consistency_weight"] = consistency_weight
            if mask_top_k > 0:
                dataset["causal_mask_top_k"] = mask_top_k
            dataset["causal_mask_min_margin"] = mask_min_margin
            if method_kind == "causal_dfr":
                method["causal_dfr_nuisance_weight"] = nuisance_weight
                method["causal_dfr_nuisance_prior"] = nuisance_prior
            config["method"] = method
            config["dataset"] = dataset
            config["name"] = f"{base.get('name', 'dfr')}_sweep"
            bundle_key = (
                str(dataset.get("path", "")),
                int(dataset.get("causal_mask_top_k", 0) or 0),
                float(dataset.get("causal_mask_min_margin", 0.0)),
            )
            if bundle_key not in bundle_cache:
                bundle_cache[bundle_key] = load_dataset(config)
            bundle = bundle_cache[bundle_key]
            metrics = evaluate(fit_method(bundle, config), bundle, config)
            row = _row(config, metrics)
            if writer is None:
                fieldnames = list(row)
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row, sort_keys=True), flush=True)
    print(json.dumps({"output": str(output), "rows": "complete"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

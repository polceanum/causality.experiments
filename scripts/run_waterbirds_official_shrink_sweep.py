from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
from pathlib import Path
import statistics
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


def _metric_row(
    *,
    row_type: str,
    label: str,
    seed: int,
    config: dict[str, Any],
    metrics: dict[str, float],
    baseline: dict[str, float] | None = None,
) -> dict[str, str]:
    dataset = dict(config.get("dataset", {}))
    method = dict(config.get("method", {}))
    baseline = baseline or {}
    test_wga = float(metrics.get("test/worst_group_accuracy", 0.0))
    baseline_test_wga = baseline.get("test/worst_group_accuracy")
    delta = "" if baseline_test_wga is None else str(test_wga - float(baseline_test_wga))
    return {
        "row_type": row_type,
        "label": label,
        "config": str(config.get("name", "")),
        "seed": str(seed),
        "method": str(method.get("kind", "")),
        "dataset_path": str(dataset.get("path", "")),
        "causal_mask_top_k": str(dataset.get("causal_mask_top_k", "")),
        "causal_mask_min_margin": str(dataset.get("causal_mask_min_margin", "")),
        "official_causal_shrink_prior": str(method.get("official_causal_shrink_prior", "")),
        "official_causal_shrink_grid": "|".join(str(value) for value in method.get("official_causal_shrink_grid", [])),
        "official_dfr_num_retrains": str(method.get("official_dfr_num_retrains", "")),
        "baseline_test_wga": "" if baseline_test_wga is None else str(baseline_test_wga),
        "test_wga": str(test_wga),
        "paired_delta_test_wga": delta,
        "val_wga": str(metrics.get("val/worst_group_accuracy", "")),
        "test_acc": str(metrics.get("test/accuracy", "")),
        "val_acc": str(metrics.get("val/accuracy", "")),
        "selected_c": str(metrics.get("model/official_dfr_best_c", "")),
        "selected_shrink": str(metrics.get("model/official_dfr_best_feature_scale", "")),
    }


def _candidate_label(config: dict[str, Any]) -> str:
    dataset = dict(config.get("dataset", {}))
    method = dict(config.get("method", {}))
    grid = "-".join(str(value) for value in method.get("official_causal_shrink_grid", []))
    return (
        f"top{dataset.get('causal_mask_top_k', '')}"
        f"_margin{dataset.get('causal_mask_min_margin', '')}"
        f"_{method.get('official_causal_shrink_prior', 'mask')}"
        f"_grid{grid}"
    ).replace(".", "p")


def _summary(
    rows: list[dict[str, str]],
    *,
    min_mean_delta: float,
    min_seed_delta: float,
    min_mean_wga: float,
) -> dict[str, Any]:
    candidates: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("row_type") != "candidate":
            continue
        candidates.setdefault(row["label"], []).append(row)
    output: dict[str, Any] = {
        "promotion_gate": {
            "min_mean_delta": min_mean_delta,
            "min_seed_delta": min_seed_delta,
            "min_mean_wga": min_mean_wga,
        },
        "candidates": [],
    }
    for label, items in sorted(candidates.items()):
        wgas = [float(item["test_wga"]) for item in items]
        deltas = [float(item["paired_delta_test_wga"]) for item in items if item["paired_delta_test_wga"]]
        mean_delta = statistics.mean(deltas) if deltas else 0.0
        min_delta = min(deltas) if deltas else 0.0
        mean_wga = statistics.mean(wgas)
        output["candidates"].append(
            {
                "label": label,
                "count": len(items),
                "mean_test_wga": mean_wga,
                "min_test_wga": min(wgas),
                "max_test_wga": max(wgas),
                "std_test_wga": statistics.pstdev(wgas) if len(wgas) > 1 else 0.0,
                "mean_delta_to_baseline": mean_delta,
                "min_delta_to_baseline": min_delta,
                "max_delta_to_baseline": max(deltas) if deltas else 0.0,
                "non_negative_seed_count": sum(delta >= 0.0 for delta in deltas),
                "passes_promotion_gate": (
                    mean_delta > min_mean_delta
                    and min_delta >= min_seed_delta
                    and mean_wga >= min_mean_wga
                ),
            }
        )
    output["candidates"].sort(
        key=lambda item: (bool(item["passes_promotion_gate"]), float(item["mean_delta_to_baseline"])),
        reverse=True,
    )
    return output


def _evaluate_config(
    config: dict[str, Any],
    bundle_cache: dict[tuple[str, int, float], Any],
) -> dict[str, float]:
    dataset = dict(config.get("dataset", {}))
    cache_key = (
        str(dataset.get("path", "")),
        int(dataset.get("causal_mask_top_k", 0) or 0),
        float(dataset.get("causal_mask_min_margin", 0.0)),
    )
    if cache_key not in bundle_cache:
        bundle_cache[cache_key] = load_dataset(config)
    bundle = bundle_cache[cache_key]
    model = fit_method(bundle, config)
    metrics = evaluate(model, bundle, config)
    details = getattr(model, "details", None)
    if isinstance(details, dict):
        for key, value in details.items():
            if isinstance(value, (int, float)):
                metrics[f"model/{key}"] = float(value)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", default="configs/benchmarks/waterbirds_features_official_dfr_val_tr_retrains50.yaml")
    parser.add_argument("--candidate-config", default="configs/benchmarks/waterbirds_features_official_causal_shrink_dfr_val_tr_gentle_retrains50.yaml")
    parser.add_argument("--dataset-path", default="data/waterbirds/features_official_erm_official_repro.csv")
    parser.add_argument("--output-csv", default="outputs/dfr_sweeps/official-shrink-paired-sweep.csv")
    parser.add_argument("--output-json", default="outputs/dfr_sweeps/official-shrink-paired-sweep.json")
    parser.add_argument("--seeds", nargs="*")
    parser.add_argument("--causal-mask-top-ks", nargs="*")
    parser.add_argument("--causal-mask-min-margins", nargs="*")
    parser.add_argument("--shrink-priors", nargs="*")
    parser.add_argument("--shrink-grid", nargs="*")
    parser.add_argument("--num-retrains", type=int, default=50)
    parser.add_argument("--min-mean-delta", type=float, default=0.003)
    parser.add_argument("--min-seed-delta", type=float, default=-0.003)
    parser.add_argument("--min-mean-wga", type=float, default=0.935)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Optional cap on hyperparameter settings; each setting still runs all requested seeds. 0 means no cap.",
    )
    args = parser.parse_args()

    baseline_base = load_config(args.baseline_config)
    candidate_base = load_config(args.candidate_config)
    baseline_base.setdefault("dataset", {})["path"] = args.dataset_path
    candidate_base.setdefault("dataset", {})["path"] = args.dataset_path
    candidate_method = dict(candidate_base.get("method", {}))
    candidate_dataset = dict(candidate_base.get("dataset", {}))

    seeds = _int_values(args.seeds, int(candidate_base.get("seed", baseline_base.get("seed", 0))))
    top_ks = _int_values(args.causal_mask_top_ks, int(candidate_dataset.get("causal_mask_top_k", 0) or 0))
    margins = _float_values(args.causal_mask_min_margins, float(candidate_dataset.get("causal_mask_min_margin", 0.0)))
    priors = _str_values(args.shrink_priors, str(candidate_method.get("official_causal_shrink_prior", "mask")))
    shrink_grid = _float_values(args.shrink_grid, 1.0)

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    bundle_cache: dict[tuple[str, int, float], Any] = {}
    rows: list[dict[str, str]] = []
    baseline_by_seed: dict[int, dict[str, float]] = {}
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer: csv.DictWriter[str] | None = None

        def write_row(row: dict[str, str]) -> None:
            nonlocal writer
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

        for seed in seeds:
            baseline_config = copy.deepcopy(baseline_base)
            baseline_config["seed"] = seed
            baseline_config["name"] = f"{baseline_base.get('name', 'official_dfr')}_seed{seed}"
            baseline_metrics = _evaluate_config(baseline_config, bundle_cache)
            baseline_by_seed[seed] = baseline_metrics
            write_row(
                _metric_row(
                    row_type="baseline",
                    label="official_dfr",
                    seed=seed,
                    config=baseline_config,
                    metrics=baseline_metrics,
                )
            )

        candidate_setting_count = 0
        for top_k, margin, prior in itertools.product(top_ks, margins, priors):
            if args.max_candidates > 0 and candidate_setting_count >= args.max_candidates:
                break
            for seed in seeds:
                config = copy.deepcopy(candidate_base)
                dataset = dict(config.get("dataset", {}))
                method = dict(config.get("method", {}))
                config["seed"] = seed
                dataset["causal_mask_top_k"] = top_k
                dataset["causal_mask_min_margin"] = margin
                method["kind"] = "official_causal_shrink_dfr_val_tr"
                method["official_dfr_num_retrains"] = int(args.num_retrains)
                method["official_causal_shrink_prior"] = prior
                method["official_causal_shrink_grid"] = shrink_grid
                config["dataset"] = dataset
                config["method"] = method
                label = _candidate_label(config)
                config["name"] = f"{candidate_base.get('name', 'official_shrink')}_{label}_seed{seed}"
                metrics = _evaluate_config(config, bundle_cache)
                write_row(
                    _metric_row(
                        row_type="candidate",
                        label=label,
                        seed=seed,
                        config=config,
                        metrics=metrics,
                        baseline=baseline_by_seed[seed],
                    )
                )
            candidate_setting_count += 1

    summary = _summary(
        rows,
        min_mean_delta=float(args.min_mean_delta),
        min_seed_delta=float(args.min_seed_delta),
        min_mean_wga=float(args.min_mean_wga),
    )
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_csv": str(output_csv), "output_json": str(output_json)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
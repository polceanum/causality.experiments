from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.clues import write_csv_rows
from causality_experiments.config import load_config
from causality_experiments.data import load_dataset
from causality_experiments.discovery import build_feature_clue_rows
from causality_experiments.run import run_experiment
from scripts.run_waterbirds_clue_fusion_sweep import (
    build_bridge_score_rows,
    build_source_score_rows,
    with_dataset_path,
    with_runtime_overrides,
)


def _float_values(values: list[str] | None, default: list[float]) -> list[float]:
    if not values:
        return default
    return [float(value) for value in values]


def _int_values(values: list[str] | None, default: list[int]) -> list[int]:
    if not values:
        return default
    return [int(value) for value in values]


def _weight_label(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _metric_payload(run_dir: Path) -> dict[str, float]:
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return dict(payload["metrics"])


def _candidate_config(
    base: dict[str, Any],
    *,
    seed: int,
    label: str,
    score_path: Path,
    top_k: int,
    nuisance_weight: float,
    dfr_num_retrains: int | None,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["seed"] = seed
    config["name"] = f"{base.get('name', 'causal_dfr')}_{label}_top{top_k}_nuis{_weight_label(nuisance_weight)}_seed{seed}"
    dataset = dict(config.get("dataset", {}))
    dataset["causal_mask_strategy"] = "discovery_scores"
    dataset["discovery_scores_path"] = str(score_path)
    dataset["discovery_score_threshold"] = 2.0
    dataset["discovery_score_top_k"] = int(top_k)
    dataset["discovery_score_soft_selection"] = "selected"
    config["dataset"] = dataset
    method = dict(config.get("method", {}))
    method["kind"] = "causal_dfr"
    method["causal_dfr_nuisance_prior"] = "soft_scores"
    method["causal_dfr_nuisance_weight"] = float(nuisance_weight)
    if dfr_num_retrains is not None:
        method["dfr_num_retrains"] = int(dfr_num_retrains)
    config["method"] = method
    return config


def _row(
    *,
    row_type: str,
    label: str,
    seed: int,
    top_k: int,
    nuisance_weight: float | None,
    run_dir: Path,
    metrics: dict[str, float],
    baseline_metrics: dict[str, float] | None,
    stats_metrics: dict[str, float] | None,
) -> dict[str, str]:
    test_wga = float(metrics.get("test/worst_group_accuracy", 0.0))
    baseline_wga = None if baseline_metrics is None else float(baseline_metrics.get("test/worst_group_accuracy", 0.0))
    stats_wga = None if stats_metrics is None else float(stats_metrics.get("test/worst_group_accuracy", 0.0))
    return {
        "row_type": row_type,
        "label": label,
        "seed": str(seed),
        "top_k": str(top_k),
        "nuisance_weight": "" if nuisance_weight is None else str(nuisance_weight),
        "run": run_dir.name,
        "val_wga": str(metrics.get("val/worst_group_accuracy", "")),
        "test_wga": str(test_wga),
        "val_acc": str(metrics.get("val/accuracy", "")),
        "test_acc": str(metrics.get("test/accuracy", "")),
        "baseline_test_wga": "" if baseline_wga is None else str(baseline_wga),
        "delta_to_baseline": "" if baseline_wga is None else str(test_wga - baseline_wga),
        "stats_test_wga": "" if stats_wga is None else str(stats_wga),
        "delta_to_stats": "" if stats_wga is None else str(test_wga - stats_wga),
    }


def _summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["row_type"] == "candidate":
            groups.setdefault(row["label"], []).append(row)
    candidates: list[dict[str, Any]] = []
    for label, items in sorted(groups.items()):
        wgas = [float(item["test_wga"]) for item in items]
        baseline_deltas = [float(item["delta_to_baseline"]) for item in items if item["delta_to_baseline"]]
        stats_deltas = [float(item["delta_to_stats"]) for item in items if item["delta_to_stats"]]
        candidates.append(
            {
                "label": label,
                "count": len(items),
                "mean_test_wga": statistics.mean(wgas),
                "min_test_wga": min(wgas),
                "max_test_wga": max(wgas),
                "mean_delta_to_baseline": statistics.mean(baseline_deltas) if baseline_deltas else 0.0,
                "min_delta_to_baseline": min(baseline_deltas) if baseline_deltas else 0.0,
                "mean_delta_to_stats": statistics.mean(stats_deltas) if stats_deltas else 0.0,
                "min_delta_to_stats": min(stats_deltas) if stats_deltas else 0.0,
                "non_negative_baseline_seeds": sum(delta >= 0.0 for delta in baseline_deltas),
                "non_negative_stats_seeds": sum(delta >= 0.0 for delta in stats_deltas),
            }
        )
    candidates.sort(key=lambda item: (item["mean_delta_to_baseline"], item["mean_delta_to_stats"]), reverse=True)
    return {"candidates": candidates}


def run_bridge_causal_dfr_sweep(
    *,
    baseline_config_path: Path,
    candidate_config_path: Path,
    dataset_path: str,
    bridge_input_dir: Path,
    out_dir: Path,
    output_csv: Path,
    output_json: Path,
    seeds: list[int],
    top_k_values: list[int],
    bridge_fused_weight: float,
    nuisance_weights: list[float],
    bridge_alpha: float,
    bridge_exclude_datasets: list[str],
    card_top_k: int,
    dfr_num_retrains: int | None,
    training_device: str | None,
    output_root: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    score_base = with_runtime_overrides(
        with_dataset_path(load_config(candidate_config_path), dataset_path),
        training_device=training_device,
    )
    bundle = load_dataset(score_base)
    clue_rows = build_feature_clue_rows(bundle, split_name="train")
    stats_path = out_dir / "scores_stats.csv"
    bridge_path = out_dir / f"scores_bridge_fused_w{_weight_label(bridge_fused_weight)}.csv"
    write_csv_rows(stats_path, build_source_score_rows(clue_rows, "stats"))
    write_csv_rows(
        bridge_path,
        build_bridge_score_rows(
            bundle,
            bridge_input_dir=bridge_input_dir,
            alpha=bridge_alpha,
            exclude_datasets=bridge_exclude_datasets,
            split_name="train",
            card_top_k=card_top_k,
            blend_with_stats_weight=bridge_fused_weight,
        ),
    )

    baseline_base = with_runtime_overrides(
        with_dataset_path(load_config(baseline_config_path), dataset_path),
        training_device=training_device,
    )
    candidate_base = with_runtime_overrides(
        with_dataset_path(load_config(candidate_config_path), dataset_path),
        training_device=training_device,
    )
    rows: list[dict[str, str]] = []
    baseline_by_seed: dict[int, dict[str, float]] = {}
    stats_by_seed_top_k_weight: dict[tuple[int, int, float], dict[str, float]] = {}

    with output_csv.open("w", encoding="utf-8", newline="") as handle, tempfile.TemporaryDirectory() as tmp_dir:
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

        tmp_root = Path(tmp_dir)
        for seed in seeds:
            baseline = copy.deepcopy(baseline_base)
            baseline["seed"] = seed
            baseline["name"] = f"{baseline.get('name', 'official_dfr')}_seed{seed}"
            config_path = tmp_root / f"{baseline['name']}.yaml"
            config_path.write_text(yaml.safe_dump(baseline, sort_keys=False), encoding="utf-8")
            run_dir = run_experiment(config_path, output_root)
            metrics = _metric_payload(run_dir)
            baseline_by_seed[seed] = metrics
            write_row(
                _row(
                    row_type="baseline",
                    label="official_dfr",
                    seed=seed,
                    top_k=0,
                    nuisance_weight=None,
                    run_dir=run_dir,
                    metrics=metrics,
                    baseline_metrics=None,
                    stats_metrics=None,
                )
            )

        for seed, top_k, nuisance_weight in itertools.product(seeds, top_k_values, nuisance_weights):
            stats_config = _candidate_config(
                candidate_base,
                seed=seed,
                label="stats",
                score_path=stats_path,
                top_k=top_k,
                nuisance_weight=nuisance_weight,
                dfr_num_retrains=dfr_num_retrains,
            )
            config_path = tmp_root / f"{stats_config['name']}.yaml"
            config_path.write_text(yaml.safe_dump(stats_config, sort_keys=False), encoding="utf-8")
            run_dir = run_experiment(config_path, output_root)
            metrics = _metric_payload(run_dir)
            stats_by_seed_top_k_weight[(seed, top_k, nuisance_weight)] = metrics
            write_row(
                _row(
                    row_type="stats_control",
                    label="stats",
                    seed=seed,
                    top_k=top_k,
                    nuisance_weight=nuisance_weight,
                    run_dir=run_dir,
                    metrics=metrics,
                    baseline_metrics=baseline_by_seed[seed],
                    stats_metrics=None,
                )
            )

        for seed, top_k, nuisance_weight in itertools.product(seeds, top_k_values, nuisance_weights):
            label = f"bridge_fused_w{_weight_label(bridge_fused_weight)}"
            config = _candidate_config(
                candidate_base,
                seed=seed,
                label=label,
                score_path=bridge_path,
                top_k=top_k,
                nuisance_weight=nuisance_weight,
                dfr_num_retrains=dfr_num_retrains,
            )
            config_path = tmp_root / f"{config['name']}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            run_dir = run_experiment(config_path, output_root)
            metrics = _metric_payload(run_dir)
            write_row(
                _row(
                    row_type="candidate",
                    label=f"{label}_top{top_k}_nuis{_weight_label(nuisance_weight)}",
                    seed=seed,
                    top_k=top_k,
                    nuisance_weight=nuisance_weight,
                    run_dir=run_dir,
                    metrics=metrics,
                    baseline_metrics=baseline_by_seed[seed],
                    stats_metrics=stats_by_seed_top_k_weight[(seed, top_k, nuisance_weight)],
                )
            )

    summary = _summary(rows)
    summary.update(
        {
            "output_csv": str(output_csv),
            "out_dir": str(out_dir),
            "seeds": seeds,
            "top_k_values": top_k_values,
            "bridge_fused_weight": bridge_fused_weight,
            "nuisance_weights": nuisance_weights,
            "bridge_alpha": bridge_alpha,
            "bridge_exclude_datasets": bridge_exclude_datasets,
            "dfr_num_retrains": dfr_num_retrains,
        }
    )
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", default="configs/benchmarks/waterbirds_features_official_dfr_val_tr_retrains50.yaml")
    parser.add_argument("--candidate-config", default="configs/benchmarks/waterbirds_features_official_causal_dfr_soft.yaml")
    parser.add_argument("--dataset-path", default="data/waterbirds/features_official_erm_official_repro.csv")
    parser.add_argument("--bridge-input-dir", default="outputs/dfr_sweeps/llm_clue_fixture_experiments")
    parser.add_argument("--out-dir", default="outputs/dfr_sweeps/bridge_causal_dfr_sweep")
    parser.add_argument("--output-csv", default="outputs/dfr_sweeps/bridge-causal-dfr-sweep.csv")
    parser.add_argument("--output-json", default="outputs/dfr_sweeps/bridge-causal-dfr-sweep.json")
    parser.add_argument("--seeds", nargs="*")
    parser.add_argument("--top-k", nargs="*")
    parser.add_argument("--bridge-fused-weight", type=float, default=0.2)
    parser.add_argument("--nuisance-weights", nargs="*")
    parser.add_argument("--bridge-alpha", type=float, default=10.0)
    parser.add_argument("--bridge-exclude-dataset", action="append", default=["waterbirds"])
    parser.add_argument("--card-top-k", type=int, default=16)
    parser.add_argument("--dfr-num-retrains", type=int, default=0)
    parser.add_argument("--training-device", default="")
    parser.add_argument("--output-root", default="outputs/runs")
    args = parser.parse_args()

    summary = run_bridge_causal_dfr_sweep(
        baseline_config_path=Path(args.baseline_config),
        candidate_config_path=Path(args.candidate_config),
        dataset_path=args.dataset_path,
        bridge_input_dir=Path(args.bridge_input_dir),
        out_dir=Path(args.out_dir),
        output_csv=Path(args.output_csv),
        output_json=Path(args.output_json),
        seeds=_int_values(args.seeds, [101]),
        top_k_values=_int_values(args.top_k, [512]),
        bridge_fused_weight=float(args.bridge_fused_weight),
        nuisance_weights=_float_values(args.nuisance_weights, [10.0, 30.0, 60.0]),
        bridge_alpha=float(args.bridge_alpha),
        bridge_exclude_datasets=list(dict.fromkeys(args.bridge_exclude_dataset)),
        card_top_k=int(args.card_top_k),
        dfr_num_retrains=int(args.dfr_num_retrains) if int(args.dfr_num_retrains) > 0 else None,
        training_device=args.training_device or None,
        output_root=Path(args.output_root),
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

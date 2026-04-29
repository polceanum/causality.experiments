from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from copy import deepcopy
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

from causality_experiments.config import load_config
from causality_experiments.run import run_experiment


def _compact_metrics(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    config = payload["config"]
    return {
        "config": config.get("name", ""),
        "method": config.get("method", {}).get("kind", ""),
        "run": run_dir.name,
        "val_wga": float(metrics.get("val/worst_group_accuracy", 0.0)),
        "test_wga": float(metrics.get("test/worst_group_accuracy", 0.0)),
        "val_acc": float(metrics.get("val/accuracy", 0.0)),
        "test_acc": float(metrics.get("test/accuracy", 0.0)),
        "probe_selectivity": float(metrics.get("probe/selectivity", 0.0)),
        "nuisance_to_causal_importance": float(metrics.get("feature_importance/nuisance_to_causal", 0.0)),
    }


def _configured_run(
    base: dict[str, Any],
    *,
    name: str,
    seed: int,
    features_csv: Path,
    output_root: Path,
    device: str,
) -> Path:
    config = deepcopy(base)
    config["name"] = name
    config["seed"] = seed
    config.setdefault("dataset", {})
    config["dataset"]["path"] = str(features_csv)
    config.setdefault("training", {})
    config["training"]["device"] = device
    config["output_dir"] = str(output_root)
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return run_experiment(config_path, output_root)


def _summary(rows: list[dict[str, Any]], baseline_label: str) -> dict[str, Any]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["label"])].append(row)
    baseline_by_seed = {
        int(row["seed"]): float(row["test_wga"])
        for row in by_label[baseline_label]
    }
    output: dict[str, Any] = {"baseline_label": baseline_label, "candidates": []}
    for label, items in sorted(by_label.items()):
        wgas = [float(item["test_wga"]) for item in items]
        accs = [float(item["test_acc"]) for item in items]
        sels = [float(item["probe_selectivity"]) for item in items]
        nuis = [float(item["nuisance_to_causal_importance"]) for item in items]
        deltas = [
            float(item["test_wga"]) - baseline_by_seed[int(item["seed"])]
            for item in items
            if int(item["seed"]) in baseline_by_seed
        ]
        output["candidates"].append(
            {
                "label": label,
                "method": items[0]["method"],
                "count": len(items),
                "mean_test_wga": statistics.mean(wgas),
                "min_test_wga": min(wgas),
                "max_test_wga": max(wgas),
                "std_test_wga": statistics.pstdev(wgas) if len(wgas) > 1 else 0.0,
                "mean_test_acc": statistics.mean(accs),
                "mean_probe_selectivity": statistics.mean(sels),
                "mean_nuisance_to_causal_importance": statistics.mean(nuis),
                "paired_deltas": deltas,
                "mean_delta_to_baseline": statistics.mean(deltas) if deltas else None,
                "non_negative_seed_count": sum(delta >= 0.0 for delta in deltas),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", default="data/waterbirds/features_official_erm_official_repro.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-csv", default="outputs/dfr_sweeps/official-representation-sweep.csv")
    parser.add_argument("--output-json", default="outputs/dfr_sweeps/official-representation-sweep.json")
    parser.add_argument("--output-root", default="outputs/runs")
    parser.add_argument("--seeds", nargs="*", type=int, default=[101, 102, 103])
    parser.add_argument(
        "--baseline-config",
        default="configs/benchmarks/waterbirds_features_official_dfr_val_tr.yaml",
    )
    parser.add_argument(
        "--candidate-configs",
        nargs="*",
        default=[
            "configs/benchmarks/waterbirds_features_official_causal_dfr.yaml",
            "configs/benchmarks/waterbirds_features_official_adv_representation_dfr.yaml",
            "configs/benchmarks/waterbirds_features_official_adv_representation_dfr_score_gate.yaml",
            "configs/benchmarks/waterbirds_features_official_adv_representation_dfr_nuisance_regularized.yaml",
        ],
    )
    args = parser.parse_args()

    features_csv = Path(args.features_csv)
    output_root = Path(args.output_root)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    baseline_base = load_config(args.baseline_config)
    baseline_label = Path(args.baseline_config).stem
    candidate_bases = [(Path(path).stem, load_config(path)) for path in args.candidate_configs]
    rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        baseline_run = _configured_run(
            baseline_base,
            name=f"{baseline_label}_seed{seed}",
            seed=seed,
            features_csv=features_csv,
            output_root=output_root,
            device=args.device,
        )
        baseline_metrics = _compact_metrics(baseline_run)
        rows.append({"label": baseline_label, "seed": seed, **baseline_metrics})
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
        for label, base in candidate_bases:
            run_dir = _configured_run(
                base,
                name=f"{label}_seed{seed}",
                seed=seed,
                features_csv=features_csv,
                output_root=output_root,
                device=args.device,
            )
            metrics = _compact_metrics(run_dir)
            rows.append({"label": label, "seed": seed, **metrics})
            print(json.dumps(rows[-1], sort_keys=True), flush=True)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(rows, baseline_label)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_csv": str(output_csv), "output_json": str(output_json)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

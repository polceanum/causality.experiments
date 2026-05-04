from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import itertools
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.component_representation import feature_columns_from_frame, infer_feature_components, load_feature_components_from_manifest
from causality_experiments.config import load_config
from causality_experiments.run import run_experiment
from scripts.build_waterbirds_component_features import build_component_feature_artifacts


def _slug(values: tuple[str, ...]) -> str:
    return "_".join(value.replace("_minus_", "minus").replace("_", "") for value in values)


def _read_metrics(run_dir: Path) -> tuple[dict[str, float], dict[str, Any]]:
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return dict(payload["metrics"]), dict(payload.get("model_details", {}))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _metric_row(
    *,
    label: str,
    components: tuple[str, ...],
    config: dict[str, Any],
    run_dir: Path,
    metrics: dict[str, float],
    details: dict[str, Any],
    raw_test_wga: float | None,
) -> dict[str, Any]:
    test_wga = float(metrics.get("test/worst_group_accuracy", 0.0))
    return {
        "label": label,
        "components": ",".join(components),
        "component_count": len(components),
        "run": run_dir.name,
        "config": str(config.get("name", "")),
        "dataset_path": str(config.get("dataset", {}).get("path", "")),
        "test_wga": test_wga,
        "delta_to_raw": "" if raw_test_wga is None else test_wga - float(raw_test_wga),
        "test_acc": float(metrics.get("test/accuracy", 0.0)),
        "val_wga": float(metrics.get("val/worst_group_accuracy", 0.0)),
        "val_acc": float(metrics.get("val/accuracy", 0.0)),
        "probe_selectivity": float(metrics.get("probe/selectivity", 0.0)),
        "nuisance_to_causal_importance": float(metrics.get("feature_importance/nuisance_to_causal", 0.0)),
        "official_dfr_best_c": details.get("official_dfr_best_c", ""),
        "official_dfr_best_tune_wga": details.get("official_dfr_best_tune_wga", ""),
    }


def _run_config(
    *,
    base_config: dict[str, Any],
    name: str,
    feature_csv: Path,
    feature_source: Path,
    split_definition: str,
    output_root: Path,
    seed: int,
    num_retrains: int,
    device: str,
) -> tuple[dict[str, Any], Path, dict[str, float], dict[str, Any]]:
    config = deepcopy(base_config)
    config["name"] = name
    config["seed"] = seed
    config.setdefault("dataset", {})["path"] = str(feature_csv)
    config.setdefault("method", {})["official_dfr_num_retrains"] = int(num_retrains)
    config.setdefault("training", {})["device"] = device
    config["output_dir"] = str(output_root)
    config.setdefault("benchmark", {})["comparable_to_literature"] = False
    config.setdefault("benchmark", {}).setdefault("provenance", {})
    config["benchmark"]["provenance"] = {
        "feature_extractor": f"component_subset_sweep/{name}",
        "feature_source": str(feature_source),
        "split_definition": split_definition,
    }
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run_dir = run_experiment(config_path, output_root)
    metrics, details = _read_metrics(run_dir)
    return config, run_dir, metrics, details


def run_component_subset_sweep(
    *,
    input_csv: Path,
    base_config_path: Path,
    out_dir: Path,
    output_csv: Path,
    output_json: Path,
    output_root: Path,
    seed: int,
    num_retrains: int,
    device: str,
    include_components: tuple[str, ...],
    exclude_components: tuple[str, ...],
    max_components: int,
    require_component: tuple[str, ...],
) -> dict[str, Any]:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_config(base_config_path)
    frame = pd.read_csv(input_csv, nrows=1)
    feature_columns = feature_columns_from_frame(frame)
    manifest_path = input_csv.with_suffix(input_csv.suffix + ".manifest.json")
    components = infer_feature_components(feature_columns, load_feature_components_from_manifest(manifest_path))
    excluded = set(exclude_components)
    original_available = [component for component in components if component not in excluded]
    available = list(original_available)
    if include_components:
        include_set = set(include_components)
        available = [component for component in available if component in include_set]
    required = tuple(component for component in require_component if component in available)
    if not available:
        raise ValueError("No components available for subset sweep.")

    raw_config, raw_run, raw_metrics, raw_details = _run_config(
        base_config=base_config,
        name=f"component_subset_raw_seed{seed}",
        feature_csv=input_csv,
        feature_source=input_csv,
        split_definition="raw component feature table",
        output_root=output_root,
        seed=seed,
        num_retrains=num_retrains,
        device=device,
    )
    raw_test_wga = float(raw_metrics.get("test/worst_group_accuracy", 0.0))
    rows = [
        _metric_row(
            label="raw_all",
            components=tuple(available),
            config=raw_config,
            run_dir=raw_run,
            metrics=raw_metrics,
            details=raw_details,
            raw_test_wga=None,
        )
    ]

    subset_max = len(available) if max_components <= 0 else min(max_components, len(available))
    seen: set[tuple[str, ...]] = set()
    for width in range(1, subset_max + 1):
        for subset in itertools.combinations(available, width):
            if set(required) - set(subset):
                continue
            if tuple(subset) == tuple(original_available):
                continue
            if subset in seen:
                continue
            seen.add(subset)
            label = _slug(subset)
            subset_csv = out_dir / f"component_subset_{label}.csv"
            build_component_feature_artifacts(
                input_csv=input_csv,
                output_csv=subset_csv,
                output_clues_csv=out_dir / f"component_subset_{label}_clues.csv",
                output_summary_csv=out_dir / f"component_subset_{label}_summary.csv",
                manifest_path=manifest_path if manifest_path.exists() else None,
                include_components=subset,
            )
            config, run_dir, metrics, details = _run_config(
                base_config=base_config,
                name=f"component_subset_{label}_seed{seed}",
                feature_csv=subset_csv,
                feature_source=input_csv,
                split_definition=f"component subset: {','.join(subset)}",
                output_root=output_root,
                seed=seed,
                num_retrains=num_retrains,
                device=device,
            )
            row = _metric_row(
                label=label,
                components=subset,
                config=config,
                run_dir=run_dir,
                metrics=metrics,
                details=details,
                raw_test_wga=raw_test_wga,
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    rows.sort(key=lambda row: float(row["test_wga"]), reverse=True)
    _write_rows(output_csv, rows)
    summary = {
        "input_csv": str(input_csv),
        "seed": seed,
        "num_retrains": num_retrains,
        "raw_test_wga": raw_test_wga,
        "component_count": len(available),
        "components": available,
        "best": rows[0],
        "rows": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--base-config", default="configs/benchmarks/waterbirds_features_official_dfr_val_tr.yaml")
    parser.add_argument("--out-dir", default="outputs/dfr_sweeps/component_subset_sweep")
    parser.add_argument("--output-csv", default="outputs/dfr_sweeps/component-subset-sweep.csv")
    parser.add_argument("--output-json", default="outputs/dfr_sweeps/component-subset-sweep.json")
    parser.add_argument("--output-root", default="outputs/runs/component_subset_sweep")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--num-retrains", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-components", nargs="*", default=[])
    parser.add_argument("--exclude-components", nargs="*", default=[])
    parser.add_argument("--require-component", nargs="*", default=[])
    parser.add_argument("--max-components", type=int, default=0)
    args = parser.parse_args()
    summary = run_component_subset_sweep(
        input_csv=Path(args.input_csv),
        base_config_path=Path(args.base_config),
        out_dir=Path(args.out_dir),
        output_csv=Path(args.output_csv),
        output_json=Path(args.output_json),
        output_root=Path(args.output_root),
        seed=args.seed,
        num_retrains=args.num_retrains,
        device=args.device,
        include_components=tuple(args.include_components),
        exclude_components=tuple(args.exclude_components),
        max_components=args.max_components,
        require_component=tuple(args.require_component),
    )
    print(json.dumps({"best": summary["best"], "raw_test_wga": summary["raw_test_wga"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

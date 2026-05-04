from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import tempfile
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.config import load_config
from causality_experiments.run import run_experiment
from scripts.build_waterbirds_component_features import build_component_feature_artifacts
from scripts.train_waterbirds_component_adapter import train_adapter_artifact


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _shuffle_adapter_priors(source: Path, output: Path, *, seed: int) -> None:
    rows = _read_rows(source)
    priors = [row.get("adapter_prior", "") for row in rows]
    rng = random.Random(seed)
    rng.shuffle(priors)
    for row, prior in zip(rows, priors, strict=True):
        row["adapter_prior"] = prior
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(output, rows)


def _metrics(run_dir: Path) -> tuple[dict[str, float], dict[str, Any]]:
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return dict(payload["metrics"]), dict(payload.get("model_details", {}))


def _run_official_dfr(
    *,
    base_config: dict[str, Any],
    name: str,
    dataset_path: Path,
    output_root: Path,
    seed: int,
    num_retrains: int,
    device: str,
) -> tuple[Path, dict[str, float], dict[str, Any]]:
    config = deepcopy(base_config)
    config["name"] = name
    config["seed"] = seed
    config.setdefault("dataset", {})["path"] = str(dataset_path)
    config.setdefault("method", {})["official_dfr_num_retrains"] = int(num_retrains)
    config.setdefault("training", {})["device"] = device
    config["output_dir"] = str(output_root)
    config.setdefault("benchmark", {})["comparable_to_literature"] = False
    config.setdefault("benchmark", {}).setdefault("provenance", {})
    config["benchmark"]["provenance"] = {
        "feature_extractor": f"component_adapter_gate/{name}",
        "feature_source": str(dataset_path),
        "split_definition": "component adapter feature table",
    }
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run_dir = run_experiment(config_path, output_root)
    metrics, details = _metrics(run_dir)
    return run_dir, metrics, details


def _row(
    *,
    label: str,
    row_type: str,
    feature_csv: Path,
    run_dir: Path,
    metrics: dict[str, float],
    raw_wga: float,
    best_random_wga: float | None = None,
) -> dict[str, Any]:
    test_wga = float(metrics.get("test/worst_group_accuracy", 0.0))
    return {
        "label": label,
        "row_type": row_type,
        "feature_csv": str(feature_csv),
        "run": run_dir.name,
        "test_wga": test_wga,
        "delta_to_raw": test_wga - raw_wga,
        "delta_to_best_random": "" if best_random_wga is None else test_wga - best_random_wga,
        "test_acc": float(metrics.get("test/accuracy", 0.0)),
        "val_wga": float(metrics.get("val/worst_group_accuracy", 0.0)),
        "val_acc": float(metrics.get("val/accuracy", 0.0)),
        "probe_selectivity": float(metrics.get("probe/selectivity", 0.0)),
        "nuisance_to_causal_importance": float(metrics.get("feature_importance/nuisance_to_causal", 0.0)),
    }


def run_component_adapter_gate(
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
    adapter_epochs: int,
    adapter_lr: float,
    env_penalty_weight: float,
    env_adversary_weight: float,
    clue_prior_weight: float,
    random_control_count: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_config(base_config_path)
    component_csv = out_dir / "component_full.csv"
    clue_csv = out_dir / "component_full_clues.csv"
    build_component_feature_artifacts(
        input_csv=input_csv,
        output_csv=component_csv,
        output_clues_csv=clue_csv,
        output_summary_csv=out_dir / "component_full_summary.csv",
        output_tests_csv=out_dir / "component_full_tests.csv",
    )
    raw_run, raw_metrics, _raw_details = _run_official_dfr(
        base_config=base_config,
        name=f"component_adapter_raw_seed{seed}",
        dataset_path=input_csv,
        output_root=output_root,
        seed=seed,
        num_retrains=num_retrains,
        device=device,
    )
    raw_wga = float(raw_metrics.get("test/worst_group_accuracy", 0.0))
    rows = [
        _row(
            label="raw",
            row_type="raw",
            feature_csv=input_csv,
            run_dir=raw_run,
            metrics=raw_metrics,
            raw_wga=raw_wga,
        )
    ]

    adapter_csv = out_dir / "component_adapter_clue.csv"
    adapter_json = out_dir / "component_adapter_clue.json"
    train_adapter_artifact(
        input_csv=component_csv,
        output_csv=adapter_csv,
        output_json=adapter_json,
        clue_csv=clue_csv,
        epochs=adapter_epochs,
        lr=adapter_lr,
        env_penalty_weight=env_penalty_weight,
        env_adversary_weight=env_adversary_weight,
        clue_prior_weight=clue_prior_weight,
        seed=seed,
    )
    adapter_run, adapter_metrics, _adapter_details = _run_official_dfr(
        base_config=base_config,
        name=f"component_adapter_clue_seed{seed}",
        dataset_path=adapter_csv,
        output_root=output_root,
        seed=seed,
        num_retrains=num_retrains,
        device=device,
    )

    random_rows: list[dict[str, Any]] = []
    for control_index in range(max(0, int(random_control_count))):
        random_clue_csv = out_dir / f"component_full_clues_random_{control_index}.csv"
        _shuffle_adapter_priors(clue_csv, random_clue_csv, seed=seed + control_index * 1009)
        random_csv = out_dir / f"component_adapter_random_{control_index}.csv"
        random_json = out_dir / f"component_adapter_random_{control_index}.json"
        train_adapter_artifact(
            input_csv=component_csv,
            output_csv=random_csv,
            output_json=random_json,
            clue_csv=random_clue_csv,
            epochs=adapter_epochs,
            lr=adapter_lr,
            env_penalty_weight=env_penalty_weight,
            env_adversary_weight=env_adversary_weight,
            clue_prior_weight=clue_prior_weight,
            seed=seed + control_index + 1,
        )
        random_run, random_metrics, _random_details = _run_official_dfr(
            base_config=base_config,
            name=f"component_adapter_random{control_index}_seed{seed}",
            dataset_path=random_csv,
            output_root=output_root,
            seed=seed,
            num_retrains=num_retrains,
            device=device,
        )
        random_rows.append(
            _row(
                label=f"random_{control_index}",
                row_type="random_control",
                feature_csv=random_csv,
                run_dir=random_run,
                metrics=random_metrics,
                raw_wga=raw_wga,
            )
        )

    best_random_wga = max((float(row["test_wga"]) for row in random_rows), default=None)
    rows.extend(random_rows)
    rows.append(
        _row(
            label="clue_adapter",
            row_type="candidate",
            feature_csv=adapter_csv,
            run_dir=adapter_run,
            metrics=adapter_metrics,
            raw_wga=raw_wga,
            best_random_wga=best_random_wga,
        )
    )
    rows.sort(key=lambda row: float(row["test_wga"]), reverse=True)
    _write_rows(output_csv, rows)
    summary = {
        "input_csv": str(input_csv),
        "seed": seed,
        "num_retrains": num_retrains,
        "adapter_epochs": adapter_epochs,
        "random_control_count": random_control_count,
        "raw_test_wga": raw_wga,
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
    parser.add_argument("--out-dir", default="outputs/dfr_sweeps/component_adapter_gate")
    parser.add_argument("--output-csv", default="outputs/dfr_sweeps/component-adapter-gate.csv")
    parser.add_argument("--output-json", default="outputs/dfr_sweeps/component-adapter-gate.json")
    parser.add_argument("--output-root", default="outputs/runs/component_adapter_gate")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--num-retrains", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--adapter-epochs", type=int, default=80)
    parser.add_argument("--adapter-lr", type=float, default=0.03)
    parser.add_argument("--env-penalty-weight", type=float, default=0.5)
    parser.add_argument("--env-adversary-weight", type=float, default=0.05)
    parser.add_argument("--clue-prior-weight", type=float, default=4.0)
    parser.add_argument("--random-control-count", type=int, default=3)
    args = parser.parse_args()
    summary = run_component_adapter_gate(
        input_csv=Path(args.input_csv),
        base_config_path=Path(args.base_config),
        out_dir=Path(args.out_dir),
        output_csv=Path(args.output_csv),
        output_json=Path(args.output_json),
        output_root=Path(args.output_root),
        seed=args.seed,
        num_retrains=args.num_retrains,
        device=args.device,
        adapter_epochs=args.adapter_epochs,
        adapter_lr=args.adapter_lr,
        env_penalty_weight=args.env_penalty_weight,
        env_adversary_weight=args.env_adversary_weight,
        clue_prior_weight=args.clue_prior_weight,
        random_control_count=args.random_control_count,
    )
    print(json.dumps({"best": summary["best"], "raw_test_wga": summary["raw_test_wga"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

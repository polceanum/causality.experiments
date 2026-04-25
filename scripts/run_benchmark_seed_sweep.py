from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.config import load_config
from causality_experiments.run import run_experiment


FIELDNAMES = [
    "seed",
    "run_dir",
    "val_wga",
    "test_wga",
    "val_acc",
    "test_acc",
    "causal_probe",
    "nuisance_probe",
    "selectivity",
]


def _load_existing_results(out_path: Path | None) -> tuple[list[dict[str, object]], set[int]]:
    if out_path is None or not out_path.exists():
        return [], set()
    with out_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    results: list[dict[str, object]] = []
    completed: set[int] = set()
    for row in rows:
        seed = int(row["seed"])
        completed.add(seed)
        results.append(
            {
                "seed": seed,
                "run_dir": row["run_dir"],
                "val_wga": float(row["val_wga"]),
                "test_wga": float(row["test_wga"]),
                "val_acc": float(row["val_acc"]),
                "test_acc": float(row["test_acc"]),
                "causal_probe": float(row["causal_probe"]),
                "nuisance_probe": float(row["nuisance_probe"]),
                "selectivity": float(row["selectivity"]),
            }
        )
    return results, completed


def _append_result(out_path: Path, row: dict[str, object]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exists = out_path.exists()
    with out_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_seed_sweep(config_path: Path, seeds: list[int], out_path: Path | None = None) -> list[dict[str, object]]:
    base = load_config(config_path)
    results, completed = _load_existing_results(out_path)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for seed in seeds:
            if seed in completed:
                continue
            config = deepcopy(base)
            config["seed"] = seed
            config["name"] = f"{base['name']}_seed{seed}"
            temp_config = tmp_path / f"{config['name']}.yaml"
            temp_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            run_dir = Path(run_experiment(temp_config))
            payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            row = {
                "seed": seed,
                "run_dir": run_dir.name,
                "val_wga": metrics.get("val/worst_group_accuracy"),
                "test_wga": metrics.get("test/worst_group_accuracy"),
                "val_acc": metrics.get("val/accuracy"),
                "test_acc": metrics.get("test/accuracy"),
                "causal_probe": metrics.get("probe/causal_accuracy"),
                "nuisance_probe": metrics.get("probe/nuisance_accuracy"),
                "selectivity": metrics.get("probe/selectivity"),
            }
            results.append(row)
            if out_path is not None:
                _append_result(out_path, row)
            print(json.dumps(row, sort_keys=True))
    return sorted(results, key=lambda row: int(row["seed"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Benchmark config to sweep.")
    parser.add_argument("--seeds", default="11,12,13", help="Comma-separated integer seeds.")
    parser.add_argument("--out", default="", help="Optional CSV path for persisted results.")
    args = parser.parse_args()

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    out_path = Path(args.out) if args.out else None
    run_seed_sweep(Path(args.config), seeds, out_path=out_path)


if __name__ == "__main__":
    main()
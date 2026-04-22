from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.config import load_config
from causality_experiments.literature import (
    benchmark_metadata,
    best_literature_wga,
    literature_rows,
)
from causality_experiments.run import summarize_runs


METHOD_SUFFIXES = (
    "_counterfactual_augmentation",
    "_counterfactual_adversarial",
    "_adversarial_probe",
    "_group_balanced_erm",
    "_group_dro",
    "_irm",
    "_jtt",
    "_erm",
)


def _experiment_name(name: str) -> str:
    for suffix in METHOD_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _config_metadata() -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    config_paths = [
        *Path("configs/experiments").glob("*.yaml"),
        *Path("configs/benchmarks").glob("*.yaml"),
    ]
    for config_path in config_paths:
        config = load_config(config_path)
        benchmark = benchmark_metadata(config)
        metadata[config["name"]] = {
            "benchmark_id": str(benchmark.get("id", "")),
            "benchmark_kind": str(benchmark.get("kind", "")),
            "literature_comparable": str(benchmark.get("comparable_to_literature", False)),
        }
    return metadata


def main() -> None:
    summary = summarize_runs("outputs/runs")
    rows = list(csv.DictReader(summary.open()))
    config_metadata = _config_metadata()
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        config = row.get("config", "")
        if (
            not config
            or "_seed" in config
            or "_irm_w" in config
            or "_w0p" in config
            or "_w1p" in config
        ):
            continue
        base_config = _experiment_name(config)
        if not row.get("benchmark_id") and base_config in config_metadata:
            row.update(config_metadata[base_config])
        row["config"] = base_config
        key = (base_config, row.get("method", ""))
        if key not in latest or row.get("run", "") > latest[key].get("run", ""):
            latest[key] = row
    print("config,benchmark_id,benchmark_kind,comparable,method,wga,literature_best_wga,note")
    for row in sorted(latest.values(), key=lambda item: (item.get("config", ""), item.get("method", ""))):
        config = row.get("config", "")
        benchmark_id = row.get("benchmark_id", "")
        comparable = row.get("literature_comparable", "").lower() == "true"
        lit_best = best_literature_wga(benchmark_id)
        note = "missing_benchmark_metadata"
        if benchmark_id:
            note = "fixture_only"
        if comparable and lit_best is not None:
            note = "compare_to_literature"
        elif literature_rows(benchmark_id):
            note = "has_literature_refs_but_this_run_is_fixture"
        print(
            ",".join(
                [
                    config,
                    benchmark_id,
                    row.get("benchmark_kind", ""),
                    str(comparable).lower(),
                    row.get("method", ""),
                    f"{float(row.get('test/worst_group_accuracy', 'nan')):.3f}",
                    "" if lit_best is None else f"{lit_best / 100.0:.3f}",
                    note,
                ]
            )
        )


if __name__ == "__main__":
    main()

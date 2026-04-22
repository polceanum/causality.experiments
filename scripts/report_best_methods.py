from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.run import summarize_runs


def _experiment_name(run: str) -> str:
    base = run
    for suffix in (
        "_counterfactual_augmentation",
        "_group_balanced_erm",
        "_group_dro",
        "_irm",
        "_jtt",
        "_erm",
    ):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def main() -> None:
    summary = summarize_runs("outputs/runs")
    rows = list(csv.DictReader(summary.open()))
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        config_name = row.get("config") or row["run"]
        key = (_experiment_name(config_name), row.get("method", ""))
        if key not in latest or row["run"] > latest[key]["run"]:
            latest[key] = row
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in latest.values():
        if row.get("method") not in {
            "erm",
            "group_balanced_erm",
            "group_dro",
            "irm",
            "jtt",
            "counterfactual_augmentation",
        }:
            continue
        config_name = row.get("config") or row["run"]
        if "_irm_w" in config_name or "_seed" in config_name:
            continue
        grouped.setdefault(_experiment_name(config_name), []).append(row)
    for experiment, items in sorted(grouped.items()):
        best = max(items, key=lambda row: float(row.get("test/worst_group_accuracy") or "nan"))
        print(
            experiment,
            f"best={best['method']}",
            f"wga={float(best['test/worst_group_accuracy']):.3f}",
            f"acc={float(best['test/accuracy']):.3f}",
            f"run={best['run']}",
        )


if __name__ == "__main__":
    main()

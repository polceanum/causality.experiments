from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.reporting import (
    REPORT_METHODS,
    experiment_name as _experiment_name,
    is_ad_hoc_config as _is_ad_hoc_config,
)
from causality_experiments.run import summarize_runs


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
        if row.get("method") not in REPORT_METHODS:
            continue
        config_name = row.get("config") or row["run"]
        if _is_ad_hoc_config(config_name, include_sweep_prefix=False, include_waterbirds_tune=False):
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

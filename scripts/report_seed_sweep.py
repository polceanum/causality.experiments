from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from math import sqrt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.run import summarize_runs


METHODS = {
    "erm",
    "group_balanced_erm",
    "group_dro",
    "irm",
    "jtt",
    "adversarial_probe",
    "counterfactual_augmentation",
}


def _experiment_name(config_name: str) -> str:
    for suffix in (
        "_counterfactual_augmentation",
        "_adversarial_probe",
        "_group_balanced_erm",
        "_group_dro",
        "_irm",
        "_jtt",
        "_erm",
    ):
        if suffix in config_name:
            return config_name.split(suffix, 1)[0]
    return config_name


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, sqrt(variance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="outputs/runs")
    parser.add_argument("--match", default="")
    args = parser.parse_args()

    summary = summarize_runs(args.runs)
    rows = list(csv.DictReader(summary.open()))
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        config_name = row.get("config") or row.get("run", "")
        key = (config_name, row.get("method", ""))
        if key not in latest or row.get("run", "") > latest[key].get("run", ""):
            latest[key] = row
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in latest.values():
        method = row.get("method", "")
        config_name = row.get("config") or row.get("run", "")
        if method not in METHODS or "_seed" not in config_name:
            continue
        experiment = _experiment_name(config_name)
        if args.match and args.match not in experiment:
            continue
        grouped[(experiment, method)].append(row)

    print("experiment,method,n,wga_mean,wga_std,acc_mean,acc_std")
    for (experiment, method), items in sorted(grouped.items()):
        wga = [float(item["test/worst_group_accuracy"]) for item in items]
        acc = [float(item["test/accuracy"]) for item in items]
        wga_mean, wga_std = _mean_std(wga)
        acc_mean, acc_std = _mean_std(acc)
        print(
            f"{experiment},{method},{len(items)},"
            f"{wga_mean:.3f},{wga_std:.3f},{acc_mean:.3f},{acc_std:.3f}"
        )


if __name__ == "__main__":
    main()

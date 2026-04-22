from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.run import summarize_runs


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


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
        if args.match and args.match not in config_name:
            continue
        if "_seed" in config_name or "_irm_w" in config_name or "_w0p" in config_name or "_w1p" in config_name:
            continue
        key = (config_name, row.get("method", ""))
        if key not in latest or row.get("run", "") > latest[key].get("run", ""):
            latest[key] = row

    print("config,method,wga,accuracy,causal_probe,nuisance_probe,selectivity")
    for row in sorted(latest.values(), key=lambda item: (item.get("config", ""), item.get("method", ""))):
        if "probe/causal_accuracy" not in row:
            continue
        causal_probe = _parse_float(row.get("probe/causal_accuracy", ""))
        nuisance_probe = _parse_float(row.get("probe/nuisance_accuracy", ""))
        selectivity = _parse_float(row.get("probe/selectivity", ""))
        if causal_probe is None or nuisance_probe is None or selectivity is None:
            continue
        print(
            ",".join(
                [
                    row.get("config", ""),
                    row.get("method", ""),
                    f"{float(row['test/worst_group_accuracy']):.3f}",
                    f"{float(row['test/accuracy']):.3f}",
                    f"{causal_probe:.3f}",
                    f"{nuisance_probe:.3f}",
                    f"{selectivity:.3f}",
                ]
            )
        )


if __name__ == "__main__":
    main()

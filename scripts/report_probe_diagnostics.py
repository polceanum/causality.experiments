from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.reporting import method_family as _method_family
from causality_experiments.run import summarize_runs


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def build_probe_rows(runs_dir: str | Path, match: str = "") -> list[dict[str, str]]:
    summary = summarize_runs(runs_dir)
    rows = list(csv.DictReader(summary.open()))
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        config_name = row.get("config") or row.get("run", "")
        if match and match not in config_name:
            continue
        if "_seed" in config_name or "_irm_w" in config_name or "_w0p" in config_name or "_w1p" in config_name:
            continue
        key = (config_name, row.get("method", ""))
        if key not in latest or row.get("run", "") > latest[key].get("run", ""):
            latest[key] = row

    probe_rows: list[dict[str, str]] = []
    for row in sorted(latest.values(), key=lambda item: (item.get("config", ""), item.get("method", ""))):
        if "probe/causal_accuracy" not in row:
            continue
        causal_probe = _parse_float(row.get("probe/causal_accuracy", ""))
        nuisance_probe = _parse_float(row.get("probe/nuisance_accuracy", ""))
        selectivity = _parse_float(row.get("probe/selectivity", ""))
        if causal_probe is None or nuisance_probe is None or selectivity is None:
            continue
        probe_rows.append(
            {
                "config": row.get("config", ""),
                "benchmark_id": row.get("benchmark_id", ""),
                "method": row.get("method", ""),
                "family": _method_family(row.get("method", "")),
                "run": row.get("run", ""),
                "wga": f"{float(row['test/worst_group_accuracy']):.3f}",
                "accuracy": f"{float(row['test/accuracy']):.3f}",
                "causal_probe": f"{causal_probe:.3f}",
                "nuisance_probe": f"{nuisance_probe:.3f}",
                "selectivity": f"{selectivity:.3f}",
            }
        )
    return probe_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="outputs/runs")
    parser.add_argument("--match", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    probe_rows = build_probe_rows(args.runs, match=args.match)
    fieldnames = [
        "config",
        "benchmark_id",
        "method",
        "family",
        "run",
        "wga",
        "accuracy",
        "causal_probe",
        "nuisance_probe",
        "selectivity",
    ]
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(probe_rows)
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(probe_rows)


if __name__ == "__main__":
    main()

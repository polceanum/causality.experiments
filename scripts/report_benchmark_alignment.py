from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.literature import best_literature_wga, literature_rows
from causality_experiments.run import summarize_runs


def main() -> None:
    summary = summarize_runs("outputs/runs")
    rows = list(csv.DictReader(summary.open()))
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
        key = (config, row.get("method", ""))
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

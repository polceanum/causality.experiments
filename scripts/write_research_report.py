from __future__ import annotations

import argparse
import csv
from collections import defaultdict
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
    "counterfactual_augmentation",
}


def _experiment_name(config_name: str) -> str:
    for suffix in (
        "_counterfactual_augmentation",
        "_group_balanced_erm",
        "_group_dro",
        "_irm",
        "_erm",
    ):
        if config_name.endswith(suffix):
            return config_name[: -len(suffix)]
    return config_name


def _metric(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="outputs/runs")
    parser.add_argument("--out", default="outputs/runs/research-report.md")
    parser.add_argument(
        "--match",
        default="",
        help="Only include experiments whose config/experiment name contains this text.",
    )
    args = parser.parse_args()

    summary = summarize_runs(args.runs)
    rows = list(csv.DictReader(summary.open()))
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        config_name = row.get("config") or row.get("run", "")
        key = (_experiment_name(config_name), row.get("method", ""))
        if key not in latest or row.get("run", "") > latest[key].get("run", ""):
            latest[key] = row
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in latest.values():
        method = row.get("method", "")
        config_name = row.get("config") or row.get("run", "")
        if method not in METHODS or "_irm_w" in config_name:
            continue
        experiment = _experiment_name(config_name)
        if args.match and args.match not in experiment:
            continue
        grouped[experiment].append(row)

    lines = [
        "# Research Report",
        "",
        "Best method is selected by test worst-group accuracy.",
        "",
        "| Experiment | Best Method | Test WGA | Test Acc | Support | ATE Proxy | Run |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for experiment, items in sorted(grouped.items()):
        best = max(items, key=lambda row: _metric(row, "test/worst_group_accuracy"))
        lines.append(
            "| "
            + " | ".join(
                [
                    experiment,
                    best.get("method", ""),
                    f"{_metric(best, 'test/worst_group_accuracy'):.3f}",
                    f"{_metric(best, 'test/accuracy'):.3f}",
                    f"{_metric(best, 'support_recovery'):.3f}",
                    f"{_metric(best, 'ate_proxy_error'):.3f}",
                    best.get("run", ""),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Method Rows", ""])
    for experiment, items in sorted(grouped.items()):
        lines.extend(
            [
                f"### {experiment}",
                "",
                "| Method | Test WGA | Test Acc | Support | ATE Proxy | Run |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in sorted(items, key=lambda item: item.get("method", "")):
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.get("method", ""),
                        f"{_metric(row, 'test/worst_group_accuracy'):.3f}",
                        f"{_metric(row, 'test/accuracy'):.3f}",
                        f"{_metric(row, 'support_recovery'):.3f}",
                        f"{_metric(row, 'ate_proxy_error'):.3f}",
                        row.get("run", ""),
                    ]
                )
                + " |"
            )
        lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

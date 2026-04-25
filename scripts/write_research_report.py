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
from causality_experiments.literature import best_literature_wga, literature_rows, literature_wga


METHODS = {
    "erm",
    "group_balanced_erm",
    "group_dro",
    "irm",
    "jtt",
    "adversarial_probe",
    "counterfactual_adversarial",
    "counterfactual_augmentation",
}

PROPOSED_METHODS = {
    "counterfactual_adversarial",
    "counterfactual_augmentation",
}


def _experiment_name(config_name: str) -> str:
    for suffix in (
        "_counterfactual_augmentation",
        "_counterfactual_adversarial",
        "_adversarial_probe",
        "_group_balanced_erm",
        "_group_dro",
        "_irm",
        "_jtt",
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


def _is_literature_comparable(row: dict[str, str]) -> bool:
    return row.get("literature_comparable", "").lower() == "true"


def _has_complete_provenance(row: dict[str, str]) -> bool:
    return row.get("benchmark_provenance_complete", "").lower() == "true"


def _is_ad_hoc_config(config_name: str) -> bool:
    return (
        "_seed" in config_name
        or "_irm_w" in config_name
        or "_w0p" in config_name
        or "_w1p" in config_name
        or config_name.startswith("sweep_")
        or config_name.startswith("waterbirds_tune_")
    )


def _method_family(method: str) -> str:
    return "proposed" if method in PROPOSED_METHODS else "baseline"


def _format_reference_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value / 100.0:.3f}"


def _delta_to_reference(row: dict[str, str], key: str, reference_percent: float | None) -> str:
    ours = _metric(row, key)
    if ours != ours or reference_percent is None:
        return ""
    return f"{ours - (reference_percent / 100.0):.3f}"


def _format_metric(row: dict[str, str], key: str) -> str:
    value = _metric(row, key)
    if value != value:
        return ""
    return f"{value:.3f}"


def _dataset_path_exists(experiment: str) -> bool | None:
    config_path = Path("configs/benchmarks") / f"{experiment}.yaml"
    if not config_path.exists():
        return None
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset = dict(config.get("dataset", {}))
    path = dataset.get("path")
    if not path:
        return None
    return Path(str(path)).expanduser().exists()


def _benchmark_provenance(experiment: str) -> dict[str, str]:
    config_path = Path("configs/benchmarks") / f"{experiment}.yaml"
    if not config_path.exists():
        return {}
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    benchmark = dict(config.get("benchmark", {}))
    return {key: str(value).strip() for key, value in dict(benchmark.get("provenance", {})).items()}


def _provenance_complete(experiment: str) -> bool | None:
    provenance = _benchmark_provenance(experiment)
    if not provenance:
        return None
    required = ("feature_extractor", "feature_source", "split_definition")
    return all(provenance.get(key, "") for key in required)


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
        if method not in METHODS or _is_ad_hoc_config(config_name):
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
        "## Gap Semantics",
        "",
        "- All gap columns are computed as `our metric - reference metric`.",
        "- For worst-group accuracy and average accuracy, positive means our number is higher than the reference and negative means it is lower.",
        "- On a real literature-comparable benchmark row, positive means better and negative means worse for these higher-is-better metrics.",
        "- On fixture-only rows, the sign only shows the direction of the development gap relative to the literature number; it is not a valid benchmark superiority claim.",
        "",
        "## Literature Context",
        "",
        "- Fixture numbers are not state-of-the-art claims.",
        "- Compare tiny fixtures only against local baselines.",
        "- For real benchmarks, refresh `docs/literature-context.md` and report published reference/SOTA numbers next to local results.",
        "- Current `05_waterbirds` is Waterbirds-style, not the real Waterbirds benchmark; its WGA is not directly comparable to published Waterbirds numbers.",
        "",
        "| Experiment | Benchmark | Comparable? | Status | Best Method | Family | Test WGA | Test Acc | Literature Best WGA | Gap to Best WGA | Run |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for experiment, items in sorted(grouped.items()):
        best = max(items, key=lambda row: _metric(row, "test/worst_group_accuracy"))
        benchmark_id = best.get("benchmark_id", "")
        lit_best = best_literature_wga(benchmark_id)
        comparable = _is_literature_comparable(best)
        status = "fixture_only"
        if comparable and _has_complete_provenance(best):
            status = "real_benchmark_ready"
        elif comparable:
            status = "blocked_missing_provenance"
        lines.append(
            "| "
            + " | ".join(
                [
                    experiment,
                    benchmark_id or "unknown",
                    "yes" if comparable else "no",
                    status,
                    best.get("method", ""),
                    _method_family(best.get("method", "")),
                    f"{_metric(best, 'test/worst_group_accuracy'):.3f}",
                    f"{_metric(best, 'test/accuracy'):.3f}",
                    "" if lit_best is None else f"{lit_best / 100.0:.3f}",
                    _delta_to_reference(best, "test/worst_group_accuracy", lit_best),
                    best.get("run", ""),
                ]
            )
            + " |"
        )

    blocked_benchmarks: list[tuple[str, str]] = []
    for config_path in sorted(Path("configs/benchmarks").glob("*.yaml")):
        experiment = config_path.stem
        if experiment in grouped:
            continue
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        benchmark = dict(config.get("benchmark", {}))
        benchmark_id = str(benchmark.get("id", ""))
        if not benchmark.get("comparable_to_literature") or not literature_rows(benchmark_id):
            continue
        blocked_benchmarks.append((experiment, benchmark_id))

    if blocked_benchmarks:
        lines.extend(["", "## Blocked Real Benchmark Comparisons", ""])
        lines.extend(
            [
                "| Experiment | Benchmark | Status | Literature Best WGA | Notes |",
                "| --- | --- | --- | ---: | --- |",
            ]
        )
        for experiment, benchmark_id in blocked_benchmarks:
            dataset_exists = _dataset_path_exists(experiment)
            provenance_complete = _provenance_complete(experiment)
            status = "no_runs_recorded"
            note = "Benchmark config exists but there are no recorded runs yet."
            if dataset_exists is False:
                status = "blocked_missing_local_data"
                note = "Local benchmark feature table is missing."
            elif provenance_complete is False:
                status = "blocked_missing_provenance"
                note = "Benchmark provenance is incomplete; set feature extractor/source fields before comparing to literature."
            lines.append(
                "| "
                + " | ".join(
                    [
                        experiment,
                        benchmark_id,
                        status,
                        _format_reference_metric(best_literature_wga(benchmark_id)),
                        note,
                    ]
                )
                + " |"
            )

    benchmark_ids = sorted({row.get("benchmark_id", "") for rows in grouped.values() for row in rows})
    lit_sections = [benchmark_id for benchmark_id in benchmark_ids if literature_rows(benchmark_id)]
    if lit_sections:
        lines.extend(["", "## Literature Reference Numbers", ""])
        for benchmark_id in lit_sections:
            lines.extend(
                [
                    f"### {benchmark_id}",
                    "",
                    "| Method | WGA | Avg Acc | Source | Notes |",
                    "| --- | ---: | ---: | --- | --- |",
                ]
            )
            for result in literature_rows(benchmark_id):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            result.method,
                            ""
                            if result.worst_group_accuracy is None
                            else f"{result.worst_group_accuracy / 100.0:.3f}",
                            ""
                            if result.average_accuracy is None
                            else f"{result.average_accuracy / 100.0:.3f}",
                            result.source,
                            result.notes,
                        ]
                    )
                    + " |"
                )
            lines.append("")

    comparable_groups = {
        experiment: items
        for experiment, items in grouped.items()
        if any(_is_literature_comparable(item) and _has_complete_provenance(item) for item in items)
    }
    fixture_groups = {
        experiment: items
        for experiment, items in grouped.items()
        if not any(_is_literature_comparable(item) and _has_complete_provenance(item) for item in items)
    }

    if comparable_groups:
        lines.extend(["", "## Direct Literature Comparisons", ""])
        for experiment, items in sorted(comparable_groups.items()):
            benchmark_id = items[0].get("benchmark_id", "")
            lines.extend(
                [
                    f"### {experiment}",
                    "",
                    "| Method | Family | Test WGA | Gap to ERM | Gap to JTT | Gap to GroupDRO | Gap to DFR | Gap to Best | Test Acc | Run |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for row in sorted(items, key=lambda item: item.get("method", "")):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            row.get("method", ""),
                            _method_family(row.get("method", "")),
                            f"{_metric(row, 'test/worst_group_accuracy'):.3f}",
                            _delta_to_reference(row, "test/worst_group_accuracy", literature_wga(benchmark_id, "erm")),
                            _delta_to_reference(row, "test/worst_group_accuracy", literature_wga(benchmark_id, "jtt")),
                            _delta_to_reference(row, "test/worst_group_accuracy", literature_wga(benchmark_id, "groupdro")),
                            _delta_to_reference(row, "test/worst_group_accuracy", literature_wga(benchmark_id, "dfr")),
                            _delta_to_reference(row, "test/worst_group_accuracy", best_literature_wga(benchmark_id)),
                            f"{_metric(row, 'test/accuracy'):.3f}",
                            row.get("run", ""),
                        ]
                    )
                    + " |"
                )
            lines.append("")

    comparable_probe_groups = {
        experiment: [
            item
            for item in items
            if _metric(item, "probe/causal_accuracy") == _metric(item, "probe/causal_accuracy")
        ]
        for experiment, items in comparable_groups.items()
    }
    comparable_probe_groups = {
        experiment: items for experiment, items in comparable_probe_groups.items() if items
    }
    if comparable_probe_groups:
        lines.extend(["", "## Representation Diagnostics", ""])
        lines.append(
            "These probe diagnostics measure whether each learned representation preserves label-relevant information while suppressing environment information on the real benchmark runs. Higher selectivity is better."
        )
        lines.append("")
        for experiment, items in sorted(comparable_probe_groups.items()):
            lines.extend(
                [
                    f"### {experiment}",
                    "",
                    "| Method | Family | Test WGA | Causal Probe | Nuisance Probe | Selectivity | Run |",
                    "| --- | --- | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for row in sorted(items, key=lambda item: item.get("method", "")):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            row.get("method", ""),
                            _method_family(row.get("method", "")),
                            _format_metric(row, "test/worst_group_accuracy"),
                            _format_metric(row, "probe/causal_accuracy"),
                            _format_metric(row, "probe/nuisance_accuracy"),
                            _format_metric(row, "probe/selectivity"),
                            row.get("run", ""),
                        ]
                    )
                    + " |"
                )
            lines.append("")

    fixture_literature_groups = {
        experiment: items
        for experiment, items in fixture_groups.items()
        if literature_rows(items[0].get("benchmark_id", ""))
    }
    if fixture_literature_groups:
        lines.extend(["", "## Non-Comparable Literature-Aligned Development Runs", ""])
        lines.append(
            "These rows are useful for mechanism checks only. They are shown next to published Waterbirds references to reveal the gap structure, not to claim benchmark parity."
        )
        lines.append("")
        for experiment, items in sorted(fixture_literature_groups.items()):
            benchmark_id = items[0].get("benchmark_id", "")
            lines.extend(
                [
                    f"### {experiment}",
                    "",
                    "| Method | Family | Test WGA | Gap to ERM | Gap to JTT | Gap to GroupDRO | Gap to DFR | Gap to Best | Test Acc | Run |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for row in sorted(items, key=lambda item: item.get("method", "")):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            row.get("method", ""),
                            _method_family(row.get("method", "")),
                            f"{_metric(row, 'test/worst_group_accuracy'):.3f}",
                            _delta_to_reference(row, "test/worst_group_accuracy", literature_wga(benchmark_id, "erm")),
                            _delta_to_reference(row, "test/worst_group_accuracy", literature_wga(benchmark_id, "jtt")),
                            _delta_to_reference(row, "test/worst_group_accuracy", literature_wga(benchmark_id, "groupdro")),
                            _delta_to_reference(row, "test/worst_group_accuracy", literature_wga(benchmark_id, "dfr")),
                            _delta_to_reference(row, "test/worst_group_accuracy", best_literature_wga(benchmark_id)),
                            f"{_metric(row, 'test/accuracy'):.3f}",
                            row.get("run", ""),
                        ]
                    )
                    + " |"
                )
            lines.append("")

    lines.extend(["", "## Development-Only Method Rows", ""])
    for experiment, items in sorted(fixture_groups.items()):
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

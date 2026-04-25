from __future__ import annotations

import argparse
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
    literature_avg_accuracy,
    literature_rows,
    literature_wga,
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

PROPOSED_METHODS = {
    "counterfactual_adversarial",
    "counterfactual_augmentation",
}


def _experiment_name(name: str) -> str:
    for suffix in METHOD_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _method_family(method: str) -> str:
    return "proposed" if method in PROPOSED_METHODS else "baseline"


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
            "benchmark_provenance_complete": str(benchmark.get("provenance_complete", False)),
            "benchmark_feature_extractor": str(benchmark.get("provenance", {}).get("feature_extractor", "")),
            "benchmark_feature_source": str(benchmark.get("provenance", {}).get("feature_source", "")),
            "benchmark_split_definition": str(benchmark.get("provenance", {}).get("split_definition", "")),
        }
    return metadata


def _is_ad_hoc_config(name: str) -> bool:
    return "_seed" in name or "_irm_w" in name or name.startswith("waterbirds_tune_")


def _safe_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _format_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def _format_literature_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value / 100.0:.3f}"


def _format_delta(ours: float | None, reference_percent: float | None) -> str:
    if ours is None or reference_percent is None:
        return ""
    return f"{ours - (reference_percent / 100.0):.3f}"


def _comparison_status(
    row: dict[str, str],
    benchmark_id: str,
    literature_comparable: bool,
) -> str:
    if not benchmark_id:
        return "missing_benchmark_metadata"
    if literature_rows(benchmark_id) and not literature_comparable:
        return "fixture_only"
    provenance_complete = row.get("benchmark_provenance_complete", "").lower() == "true"
    if literature_comparable and not provenance_complete:
        return "blocked_missing_provenance"
    if literature_comparable:
        return "real_benchmark_ready"
    return "no_literature_reference"


def build_alignment_rows(
    runs_dir: str | Path,
    config_dirs: tuple[str, ...] = ("configs/experiments", "configs/benchmarks"),
) -> list[dict[str, str]]:
    summary = summarize_runs(runs_dir)
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

    output_rows: list[dict[str, str]] = []
    for row in sorted(latest.values(), key=lambda item: (item.get("config", ""), item.get("method", ""))):
        benchmark_id = row.get("benchmark_id", "")
        comparable = row.get("literature_comparable", "").lower() == "true"
        test_wga = _safe_float(row.get("test/worst_group_accuracy"))
        test_acc = _safe_float(row.get("test/accuracy"))
        erm_wga = literature_wga(benchmark_id, "erm")
        jtt_wga = literature_wga(benchmark_id, "jtt")
        group_dro_wga = literature_wga(benchmark_id, "groupdro")
        dfr_wga = literature_wga(benchmark_id, "dfr")
        sota_wga = best_literature_wga(benchmark_id)
        output_rows.append(
            {
                "config": row.get("config", ""),
                "benchmark_id": benchmark_id,
                "benchmark_kind": row.get("benchmark_kind", ""),
                "comparable": str(comparable).lower(),
                "comparison_status": _comparison_status(row, benchmark_id, comparable),
                "method_family": _method_family(row.get("method", "")),
                "provenance_complete": row.get("benchmark_provenance_complete", ""),
                "feature_extractor": row.get("benchmark_feature_extractor", ""),
                "feature_source": row.get("benchmark_feature_source", ""),
                "split_definition": row.get("benchmark_split_definition", ""),
                "method": row.get("method", ""),
                "run": row.get("run", ""),
                "our_wga": _format_metric(test_wga),
                "our_acc": _format_metric(test_acc),
                "literature_erm_wga": _format_literature_metric(erm_wga),
                "literature_jtt_wga": _format_literature_metric(jtt_wga),
                "literature_groupdro_wga": _format_literature_metric(group_dro_wga),
                "literature_dfr_wga": _format_literature_metric(dfr_wga),
                "literature_sota_wga": _format_literature_metric(sota_wga),
                "delta_to_erm_wga": _format_delta(test_wga, erm_wga),
                "delta_to_jtt_wga": _format_delta(test_wga, jtt_wga),
                "delta_to_groupdro_wga": _format_delta(test_wga, group_dro_wga),
                "delta_to_dfr_wga": _format_delta(test_wga, dfr_wga),
                "delta_to_sota_wga": _format_delta(test_wga, sota_wga),
                "literature_erm_acc": _format_literature_metric(literature_avg_accuracy(benchmark_id, "erm")),
                "literature_jtt_acc": _format_literature_metric(literature_avg_accuracy(benchmark_id, "jtt")),
                "literature_groupdro_acc": _format_literature_metric(literature_avg_accuracy(benchmark_id, "groupdro")),
                "literature_dfr_acc": _format_literature_metric(literature_avg_accuracy(benchmark_id, "dfr")),
            }
        )

    known_configs = set(item[0] for item in latest)
    for config_dir in config_dirs:
        for config_path in sorted(Path(config_dir).glob("*.yaml")):
            config = load_config(config_path)
            benchmark = benchmark_metadata(config)
            config_name = str(config.get("name", ""))
            if config_name in known_configs:
                continue
            benchmark_id = str(benchmark.get("id", ""))
            comparable = bool(benchmark.get("comparable_to_literature", False))
            if not comparable or not literature_rows(benchmark_id):
                continue
            dataset = dict(config.get("dataset", {}))
            dataset_path = str(dataset.get("path", ""))
            dataset_exists = Path(dataset_path).expanduser().exists() if dataset_path else False
            provenance_complete = bool(benchmark.get("provenance_complete", False))
            status = "no_runs_recorded"
            if dataset_path and not dataset_exists:
                status = "blocked_missing_local_data"
            elif not provenance_complete:
                status = "blocked_missing_provenance"
            output_rows.append(
                {
                    "config": config_name,
                    "benchmark_id": benchmark_id,
                    "benchmark_kind": str(benchmark.get("kind", "")),
                    "comparable": str(comparable).lower(),
                    "comparison_status": status,
                    "method_family": "",
                    "provenance_complete": str(provenance_complete).lower(),
                    "feature_extractor": str(benchmark.get("provenance", {}).get("feature_extractor", "")),
                    "feature_source": str(benchmark.get("provenance", {}).get("feature_source", "")),
                    "split_definition": str(benchmark.get("provenance", {}).get("split_definition", "")),
                    "method": "",
                    "run": "",
                    "our_wga": "",
                    "our_acc": "",
                    "literature_erm_wga": _format_literature_metric(literature_wga(benchmark_id, "erm")),
                    "literature_jtt_wga": _format_literature_metric(literature_wga(benchmark_id, "jtt")),
                    "literature_groupdro_wga": _format_literature_metric(literature_wga(benchmark_id, "groupdro")),
                    "literature_dfr_wga": _format_literature_metric(literature_wga(benchmark_id, "dfr")),
                    "literature_sota_wga": _format_literature_metric(best_literature_wga(benchmark_id)),
                    "delta_to_erm_wga": "",
                    "delta_to_jtt_wga": "",
                    "delta_to_groupdro_wga": "",
                    "delta_to_dfr_wga": "",
                    "delta_to_sota_wga": "",
                    "literature_erm_acc": _format_literature_metric(literature_avg_accuracy(benchmark_id, "erm")),
                    "literature_jtt_acc": _format_literature_metric(literature_avg_accuracy(benchmark_id, "jtt")),
                    "literature_groupdro_acc": _format_literature_metric(literature_avg_accuracy(benchmark_id, "groupdro")),
                    "literature_dfr_acc": _format_literature_metric(literature_avg_accuracy(benchmark_id, "dfr")),
                }
            )
    return sorted(output_rows, key=lambda item: (item["config"], item["method"], item["comparison_status"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="outputs/runs")
    args = parser.parse_args()

    fieldnames = [
        "config",
        "benchmark_id",
        "benchmark_kind",
        "comparable",
        "comparison_status",
        "method_family",
        "provenance_complete",
        "feature_extractor",
        "feature_source",
        "split_definition",
        "method",
        "run",
        "our_wga",
        "our_acc",
        "literature_erm_wga",
        "literature_jtt_wga",
        "literature_groupdro_wga",
        "literature_dfr_wga",
        "literature_sota_wga",
        "delta_to_erm_wga",
        "delta_to_jtt_wga",
        "delta_to_groupdro_wga",
        "delta_to_dfr_wga",
        "delta_to_sota_wga",
        "literature_erm_acc",
        "literature_jtt_acc",
        "literature_groupdro_acc",
        "literature_dfr_acc",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(build_alignment_rows(args.runs))


if __name__ == "__main__":
    main()

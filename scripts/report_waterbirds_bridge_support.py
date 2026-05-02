from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _top_features(score_path: Path, top_k: int) -> list[dict[str, str]]:
    rows = _read_rows(score_path)
    return sorted(rows, key=lambda row: _safe_float(row.get("score")), reverse=True)[:top_k]


def _clue_map(clue_path: Path) -> dict[str, dict[str, str]]:
    return {row["feature_name"]: row for row in _read_rows(clue_path) if row.get("feature_name")}


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _support_summary(label: str, selected: list[dict[str, str]], clues: dict[str, dict[str, str]]) -> dict[str, Any]:
    names = [row["feature_name"] for row in selected]
    clue_rows = [clues.get(name, {}) for name in names]
    scores = [_safe_float(row.get("score")) for row in selected]
    label_corr = [_safe_float(row.get("label_corr")) for row in clue_rows]
    env_corr = [_safe_float(row.get("env_corr")) for row in clue_rows]
    corr_margin = [_safe_float(row.get("corr_margin")) for row in clue_rows]
    high_env_count = sum(env >= label for label, env in zip(label_corr, env_corr, strict=True))
    return {
        "label": label,
        "selected_count": len(names),
        "mean_score": _mean(scores),
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "mean_label_corr": _mean(label_corr),
        "mean_env_corr": _mean(env_corr),
        "mean_corr_margin": _mean(corr_margin),
        "env_ge_label_count": high_env_count,
        "features": names,
    }


def build_support_report(
    *,
    clue_path: Path,
    score_paths: dict[str, Path],
    top_k: int,
    reference_label: str,
) -> dict[str, Any]:
    clues = _clue_map(clue_path)
    selected_by_label = {label: _top_features(path, top_k) for label, path in score_paths.items()}
    selected_names = {
        label: {row["feature_name"] for row in selected}
        for label, selected in selected_by_label.items()
    }
    reference = selected_names.get(reference_label, set())
    summaries: list[dict[str, Any]] = []
    for label, selected in selected_by_label.items():
        names = selected_names[label]
        union = names | reference
        summary = _support_summary(label, selected, clues)
        summary.update(
            {
                "overlap_with_reference": len(names & reference),
                "jaccard_with_reference": 0.0 if not union else len(names & reference) / len(union),
                "only_vs_reference_count": len(names - reference),
                "missing_from_reference_count": len(reference - names),
            }
        )
        summaries.append(summary)
    summaries.sort(key=lambda row: row["label"])
    return {
        "clue_path": str(clue_path),
        "top_k": int(top_k),
        "reference_label": reference_label,
        "score_paths": {label: str(path) for label, path in score_paths.items()},
        "supports": summaries,
    }


def _parse_score_arg(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--score must use LABEL=PATH format.")
        label, path = value.split("=", 1)
        output[label.strip()] = Path(path.strip())
    if not output:
        raise ValueError("At least one --score LABEL=PATH is required.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clues", default="outputs/dfr_sweeps/bridge_fused_downstream_official_shrink_screen/merged_clues.csv")
    parser.add_argument("--score", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--reference-label", default="stats")
    parser.add_argument("--output-json", default="outputs/dfr_sweeps/bridge-fused-support-report.json")
    args = parser.parse_args()
    score_paths = _parse_score_arg(args.score) if args.score else {
        "stats": Path("outputs/dfr_sweeps/bridge_fused_refreshed_random_controls/scores_stats.csv"),
        "bridge_fused_w0p3": Path("outputs/dfr_sweeps/bridge_fused_refreshed_random_controls/scores_bridge_fused_w0p3.csv"),
        "random_score_0": Path("outputs/dfr_sweeps/bridge_fused_refreshed_random_controls/scores_random_control_0.csv"),
        "random_score_1": Path("outputs/dfr_sweeps/bridge_fused_refreshed_random_controls/scores_random_control_1.csv"),
        "random_score_2": Path("outputs/dfr_sweeps/bridge_fused_refreshed_random_controls/scores_random_control_2.csv"),
    }
    report = build_support_report(
        clue_path=Path(args.clues),
        score_paths=score_paths,
        top_k=int(args.top_k),
        reference_label=str(args.reference_label),
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output_json, flush=True)


if __name__ == "__main__":
    main()

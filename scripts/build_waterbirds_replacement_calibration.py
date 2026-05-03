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


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _top_rows(path: Path, top_k: int) -> list[dict[str, str]]:
    rows = _read_rows(path)
    return sorted(rows, key=lambda row: _safe_float(row.get("score")), reverse=True)[:top_k]


def _clue_map(path: Path) -> dict[str, dict[str, str]]:
    return {row["feature_name"]: row for row in _read_rows(path) if row.get("feature_name")}


def _feature_stats(names: set[str], clues: dict[str, dict[str, str]]) -> dict[str, float]:
    rows = [clues.get(name, {}) for name in sorted(names)]
    label_corr = [_safe_float(row.get("label_corr")) for row in rows]
    env_corr = [_safe_float(row.get("env_corr")) for row in rows]
    corr_margin = [_safe_float(row.get("corr_margin")) for row in rows]
    return {
        "count": float(len(names)),
        "env_ge_label_count": float(sum(env >= label for label, env in zip(label_corr, env_corr, strict=True))),
        "mean_label_corr": _mean(label_corr),
        "mean_env_corr": _mean(env_corr),
        "mean_corr_margin": _mean(corr_margin),
    }


def _best_random_by_seed_top_k(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    best: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.get("row_type") != "random_control":
            continue
        key = (row.get("seed", ""), row.get("top_k", ""))
        best[key] = max(best.get(key, float("-inf")), _safe_float(row.get("test_wga")))
    return best


def _outcome_summary(rows: list[dict[str, str]], label: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("row_type") == "candidate" and row.get("label") == label]
    best_random = _best_random_by_seed_top_k(rows)
    wgas = [_safe_float(row.get("test_wga")) for row in selected]
    baseline_deltas = [_safe_float(row.get("delta_to_baseline")) for row in selected if row.get("delta_to_baseline")]
    stats_deltas = [_safe_float(row.get("delta_to_stats")) for row in selected if row.get("delta_to_stats")]
    random_deltas = [
        _safe_float(row.get("test_wga")) - best_random[(row.get("seed", ""), row.get("top_k", ""))]
        for row in selected
        if (row.get("seed", ""), row.get("top_k", "")) in best_random
    ]
    return {
        "outcome_count": len(selected),
        "mean_test_wga": _mean(wgas),
        "min_test_wga": min(wgas) if wgas else 0.0,
        "mean_delta_to_baseline": _mean(baseline_deltas),
        "min_delta_to_baseline": min(baseline_deltas) if baseline_deltas else 0.0,
        "mean_delta_to_stats": _mean(stats_deltas),
        "min_delta_to_stats": min(stats_deltas) if stats_deltas else 0.0,
        "mean_delta_to_best_random": _mean(random_deltas),
        "min_delta_to_best_random": min(random_deltas) if random_deltas else 0.0,
        "non_negative_baseline_seeds": sum(delta >= 0.0 for delta in baseline_deltas),
        "non_negative_stats_seeds": sum(delta >= 0.0 for delta in stats_deltas),
        "non_negative_best_random_seeds": sum(delta >= 0.0 for delta in random_deltas),
    }


def _pair_role_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("active_boundary_pair_role") == "accepted"]
    evicted = [row for row in rows if row.get("active_boundary_pair_role") == "evicted"]
    accepted_delta = [_safe_float(row.get("active_boundary_pair_delta")) for row in accepted]
    return {
        "accepted_pair_count": len(accepted),
        "evicted_pair_count": len(evicted),
        "mean_accepted_pair_delta": _mean(accepted_delta),
        "max_accepted_pair_delta": max(accepted_delta) if accepted_delta else 0.0,
    }


def _flatten(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def build_calibration_rows(
    *,
    clue_path: Path,
    reference_score_path: Path,
    candidates: dict[str, tuple[Path, Path]],
    top_k: int,
) -> list[dict[str, Any]]:
    clues = _clue_map(clue_path)
    reference_rows = _top_rows(reference_score_path, top_k)
    reference_names = {row["feature_name"] for row in reference_rows}
    reference_stats = _feature_stats(reference_names, clues)
    rows: list[dict[str, Any]] = []
    for label, (score_path, outcome_path) in sorted(candidates.items()):
        score_rows = _read_rows(score_path)
        selected_rows = sorted(score_rows, key=lambda row: _safe_float(row.get("score")), reverse=True)[:top_k]
        selected_names = {row["feature_name"] for row in selected_rows}
        entered = selected_names - reference_names
        left = reference_names - selected_names
        union = selected_names | reference_names
        outcome_rows = _read_rows(outcome_path)
        outcome = _outcome_summary(outcome_rows, label)
        changed_count = len(entered)
        candidate_row: dict[str, Any] = {
            "label": label,
            "score_path": str(score_path),
            "outcome_path": str(outcome_path),
            "top_k": int(top_k),
            "selected_count": len(selected_names),
            "reference_selected_count": len(reference_names),
            "overlap_with_reference": len(selected_names & reference_names),
            "changed_count": changed_count,
            "changed_fraction": changed_count / max(int(top_k), 1),
            "jaccard_with_reference": 0.0 if not union else len(selected_names & reference_names) / len(union),
        }
        candidate_row.update(_flatten("reference", reference_stats))
        candidate_row.update(_flatten("candidate", _feature_stats(selected_names, clues)))
        candidate_row.update(_flatten("entered", _feature_stats(entered, clues)))
        candidate_row.update(_flatten("left", _feature_stats(left, clues)))
        candidate_row.update(_pair_role_summary(score_rows))
        candidate_row.update(outcome)
        candidate_row["clears_baseline_gate"] = int(outcome["outcome_count"] > 0 and outcome["non_negative_baseline_seeds"] == outcome["outcome_count"])
        candidate_row["clears_stats_gate"] = int(outcome["outcome_count"] > 0 and outcome["non_negative_stats_seeds"] == outcome["outcome_count"])
        candidate_row["clears_best_random_gate"] = int(
            outcome["outcome_count"] > 0
            and outcome["non_negative_best_random_seeds"] == outcome["outcome_count"]
        )
        returnable = {key: (int(value) if isinstance(value, bool) else value) for key, value in candidate_row.items()}
        rows.append(returnable)
    return rows


def _parse_candidate(values: list[str]) -> dict[str, tuple[Path, Path]]:
    parsed: dict[str, tuple[Path, Path]] = {}
    for value in values:
        parts = value.split("=")
        if len(parts) != 3:
            raise ValueError("--candidate must use LABEL=SCORE_CSV=OUTCOME_CSV format.")
        label, score_path, outcome_path = parts
        parsed[label.strip()] = (Path(score_path.strip()), Path(outcome_path.strip()))
    if not parsed:
        raise ValueError("At least one --candidate is required.")
    return parsed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clues", default="outputs/dfr_sweeps/bridge_fused_downstream_official_shrink_screen/merged_clues.csv")
    parser.add_argument("--reference-score", default="outputs/dfr_sweeps/bridge_fused_refreshed_random_controls/scores_bridge_fused_w0p3.csv")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--output-csv", default="outputs/dfr_sweeps/waterbirds-replacement-calibration.csv")
    parser.add_argument("--output-json", default="outputs/dfr_sweeps/waterbirds-replacement-calibration.json")
    args = parser.parse_args()
    candidates = _parse_candidate(args.candidate)
    rows = build_calibration_rows(
        clue_path=Path(args.clues),
        reference_score_path=Path(args.reference_score),
        candidates=candidates,
        top_k=int(args.top_k),
    )
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    write_csv(output_csv, rows)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_csv": str(output_csv), "output_json": str(output_json), "rows": len(rows)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_variant_paths(values: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        variant, sep, path_text = value.partition("=")
        if not sep or not variant.strip() or not path_text.strip():
            raise ValueError(f"Expected VARIANT=PATH, got {value!r}.")
        mapping[variant.strip().lower()] = Path(path_text.strip())
    return mapping


def _load_scored_features(path: Path, top_k: int) -> set[str]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    sorted_rows = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
    return {row["feature_name"] for row in sorted_rows[:top_k]}


def _normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi <= lo + 1e-12:
        return {key: 1.0 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def annotate_rows(
    clue_rows: list[dict[str, str]],
    result_rows: list[dict[str, str]],
    score_paths: dict[str, Path],
    *,
    metric_key: str,
    variant_key: str,
) -> list[dict[str, str]]:
    feature_values = {row["feature_name"]: 0.0 for row in clue_rows}
    feature_counts = {row["feature_name"]: 0 for row in clue_rows}
    run_scores: dict[tuple[str, int], float] = {}
    selected_features: dict[tuple[str, int], set[str]] = {}
    for row in result_rows:
        variant = str(row.get(variant_key, "")).strip().lower()
        if not variant:
            continue
        if variant not in score_paths:
            continue
        top_k = int(float(row["top_k"]))
        metric_value = float(row[metric_key])
        key = (variant, top_k)
        run_scores[key] = metric_value
        if key not in selected_features:
            selected_features[key] = _load_scored_features(score_paths[variant], top_k)
    normalized_run_scores = _normalize({f"{variant}:{top_k}": score for (variant, top_k), score in run_scores.items()})
    for (variant, top_k), features in selected_features.items():
        normalized_score = normalized_run_scores[f"{variant}:{top_k}"]
        for feature_name in features:
            if feature_name not in feature_values:
                continue
            feature_values[feature_name] += normalized_score
            feature_counts[feature_name] += 1
    max_count = max(feature_counts.values(), default=0)
    updated_rows: list[dict[str, str]] = []
    for row in clue_rows:
        feature_name = row["feature_name"]
        count = feature_counts[feature_name]
        updated = dict(row)
        if count == 0:
            updated["utility_target"] = ""
            updated["utility_value"] = ""
            updated["utility_weight"] = "0.000000"
            updated["utility_count"] = "0"
        else:
            utility_value = feature_values[feature_name] / count
            utility_weight = count / max(max_count, 1)
            updated["utility_target"] = f"{utility_value:.6f}"
            updated["utility_value"] = f"{utility_value:.6f}"
            updated["utility_weight"] = f"{utility_weight:.6f}"
            updated["utility_count"] = str(count)
        updated_rows.append(updated)
    return updated_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clues", required=True, help="Input discovery clue CSV.")
    parser.add_argument("--results", required=True, help="Candidate result CSV with top_k and variant/support column.")
    parser.add_argument(
        "--score-path",
        action="append",
        required=True,
        help="Variant to score csv mapping, e.g. full=outputs/runs/scores.csv",
    )
    parser.add_argument("--metric", default="val_wga", help="Metric column used to define downstream utility.")
    parser.add_argument(
        "--variant-column",
        default="support",
        help="Column in the results CSV that identifies which score file to use.",
    )
    parser.add_argument("--out", required=True, help="Output CSV with utility annotations.")
    args = parser.parse_args()

    clue_rows = list(csv.DictReader(Path(args.clues).open("r", encoding="utf-8", newline="")))
    result_rows = list(csv.DictReader(Path(args.results).open("r", encoding="utf-8", newline="")))
    annotated = annotate_rows(
        clue_rows,
        result_rows,
        _parse_variant_paths(args.score_path),
        metric_key=args.metric,
        variant_key=args.variant_column,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(annotated[0].keys()))
        writer.writeheader()
        writer.writerows(annotated)
    print(out_path)


if __name__ == "__main__":
    main()
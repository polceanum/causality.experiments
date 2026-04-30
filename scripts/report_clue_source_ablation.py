from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.clues import write_csv_rows


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _feature_rows(path: Path) -> dict[str, dict[str, str]]:
    return {
        str(row.get("feature_name", "")).strip(): row
        for row in _read_rows(path)
        if str(row.get("feature_name", "")).strip()
    }


def _safe_float(row: dict[str, Any], key: str) -> float:
    value = str(row.get(key, "")).strip()
    if not value or value.lower() == "nan":
        return 0.0
    return float(value)


def _mean(rows: list[dict[str, str]], key: str) -> str:
    if not rows:
        return ""
    return f"{statistics.fmean(_safe_float(row, key) for row in rows):.6f}"


def _selected_scores(path: Path, top_k: int) -> list[dict[str, str]]:
    rows = [row for row in _read_rows(path) if str(row.get("feature_name", "")).strip()]
    return sorted(rows, key=lambda row: _safe_float(row, "score"), reverse=True)[:top_k]


def summarize_source_ablation(
    clue_path: Path,
    score_specs: list[tuple[str, Path]],
    *,
    top_k_values: list[int],
    reference_label: str | None = None,
) -> list[dict[str, str]]:
    if not top_k_values:
        raise ValueError("At least one top-k value is required.")
    clue_rows = _feature_rows(clue_path)
    if not clue_rows:
        raise ValueError(f"No clue rows found in {clue_path}.")
    selected_by_label: dict[tuple[str, int], set[str]] = {}
    rows: list[dict[str, str]] = []
    for label, score_path in score_specs:
        for top_k in top_k_values:
            if top_k <= 0:
                raise ValueError("top-k values must be positive.")
            selected_scores = _selected_scores(score_path, top_k)
            selected_names = [str(row["feature_name"]) for row in selected_scores if str(row.get("feature_name", "")) in clue_rows]
            if not selected_names:
                raise ValueError(f"No overlapping features found for {label} at top_k={top_k}.")
            selected_by_label[(label, top_k)] = set(selected_names)
            joined = [clue_rows[name] for name in selected_names]
            row = {
                "label": label,
                "top_k": str(len(joined)),
                "mean_score": f"{statistics.fmean(_safe_float(row, 'score') for row in selected_scores[: len(joined)]):.6f}",
                "mean_label_corr": _mean(joined, "label_corr"),
                "mean_env_corr": _mean(joined, "env_corr"),
                "mean_corr_margin": _mean(joined, "corr_margin"),
                "mean_abs_margin": f"{statistics.fmean(abs(_safe_float(row, 'corr_margin')) for row in joined):.6f}",
                "mean_language_causal_score": _mean(joined, "language_causal_score"),
                "mean_language_spurious_score": _mean(joined, "language_spurious_score"),
                "mean_language_confidence": _mean(joined, "language_confidence"),
                "mean_image_label_score": _mean(joined, "image_label_score"),
                "mean_image_background_score": _mean(joined, "image_background_score"),
                "mean_image_confidence": _mean(joined, "image_confidence"),
                "mean_top_activation_group_entropy": _mean(joined, "top_activation_group_entropy"),
                "mean_label_env_disentanglement": _mean(joined, "label_env_disentanglement"),
                "overlap_reference": "",
                "jaccard_reference": "",
            }
            rows.append(row)

    if reference_label:
        for row in rows:
            top_k = int(row["top_k"])
            selected = selected_by_label.get((row["label"], top_k), set())
            reference = selected_by_label.get((reference_label, top_k), set())
            if reference:
                overlap = selected & reference
                row["overlap_reference"] = str(len(overlap))
                row["jaccard_reference"] = f"{len(overlap) / max(len(selected | reference), 1):.6f}"
    return rows


def _parse_score_spec(value: str) -> tuple[str, Path]:
    label, sep, path_text = value.partition("=")
    if not sep or not label.strip() or not path_text.strip():
        raise ValueError("Each --scores value must be label=path/to/scores.csv")
    return label.strip(), Path(path_text.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clues", required=True, help="Merged clue row CSV.")
    parser.add_argument("--scores", action="append", required=True, help="Score spec label=path. Can be passed multiple times.")
    parser.add_argument("--top-k", type=int, action="append", required=True, help="Top-k value. Can be passed multiple times.")
    parser.add_argument("--reference-label", default="", help="Optional label to use for overlap/Jaccard columns.")
    parser.add_argument("--out", default="", help="Optional CSV path for source-ablation rows.")
    args = parser.parse_args()

    rows = summarize_source_ablation(
        Path(args.clues),
        [_parse_score_spec(value) for value in args.scores],
        top_k_values=args.top_k,
        reference_label=args.reference_label or None,
    )
    if args.out:
        write_csv_rows(Path(args.out), rows)

    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()

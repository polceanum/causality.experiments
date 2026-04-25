from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import sys


def _load_clue_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row["feature_name"]): row for row in rows if str(row.get("feature_name", "")).strip()}


def _load_scores(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if str(row.get("feature_name", "")).strip()]


def _safe_float(row: dict[str, str], key: str) -> float:
    value = str(row.get(key, "")).strip()
    return float(value) if value else 0.0


def summarize_topk_clue_quality(
    clue_path: Path,
    score_path: Path,
    *,
    top_k: int,
    label: str,
) -> dict[str, str]:
    clue_rows = _load_clue_rows(clue_path)
    scored = sorted(_load_scores(score_path), key=lambda row: _safe_float(row, "score"), reverse=True)
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    selected = [row for row in scored if row["feature_name"] in clue_rows][:top_k]
    if not selected:
        raise ValueError(f"No overlapping features found between {score_path} and {clue_path}.")

    joined = [clue_rows[row["feature_name"]] for row in selected]
    label_corr = [_safe_float(row, "label_corr") for row in joined]
    env_corr = [_safe_float(row, "env_corr") for row in joined]
    margins = [_safe_float(row, "corr_margin") for row in joined]
    support = [_safe_float(row, "causal_target") for row in joined if str(row.get("causal_target", "")).strip().lower() != "nan"]
    explicit = [_safe_float(row, "supervision_explicit") for row in joined]

    return {
        "label": label,
        "top_k": str(len(joined)),
        "mean_label_corr": f"{statistics.fmean(label_corr):.6f}",
        "mean_env_corr": f"{statistics.fmean(env_corr):.6f}",
        "mean_corr_margin": f"{statistics.fmean(margins):.6f}",
        "mean_abs_margin": f"{statistics.fmean(abs(value) for value in margins):.6f}",
        "mean_causal_target": f"{statistics.fmean(support):.6f}" if support else "",
        "mean_explicit_supervision": f"{statistics.fmean(explicit):.6f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clues", required=True, help="CSV of feature clue rows with feature_name and clue statistics.")
    parser.add_argument("--scores", action="append", required=True, help="Score spec in the form label=path/to/scores.csv. Can be passed multiple times.")
    parser.add_argument("--top-k", type=int, action="append", required=True, help="Top-k values to summarize. Can be passed multiple times.")
    parser.add_argument("--out", default="", help="Optional CSV path for the summary rows.")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    clue_path = Path(args.clues)
    for score_spec in args.scores:
        label, sep, value = score_spec.partition("=")
        if not sep or not label.strip() or not value.strip():
            raise ValueError("Each --scores value must be label=path/to/scores.csv")
        score_path = Path(value.strip())
        for top_k in args.top_k:
            rows.append(
                summarize_topk_clue_quality(
                    clue_path,
                    score_path,
                    top_k=top_k,
                    label=label.strip(),
                )
            )

    fieldnames = list(rows[0].keys())
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
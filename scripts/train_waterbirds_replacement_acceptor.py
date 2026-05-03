from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np


FEATURE_COLUMNS = (
    "changed_fraction",
    "jaccard_with_reference",
    "candidate_env_ge_label_count",
    "entered_count",
    "entered_env_ge_label_count",
    "entered_mean_label_corr",
    "entered_mean_env_corr",
    "entered_mean_corr_margin",
    "left_count",
    "left_env_ge_label_count",
    "left_mean_label_corr",
    "left_mean_env_corr",
    "left_mean_corr_margin",
    "accepted_pair_count",
    "mean_accepted_pair_delta",
    "max_accepted_pair_delta",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _target(row: dict[str, str]) -> float:
    return min(
        _safe_float(row.get("mean_delta_to_stats")),
        _safe_float(row.get("mean_delta_to_best_random")),
    )


def _feature_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [[_safe_float(row.get(column)) for column in FEATURE_COLUMNS] for row in rows],
        dtype=np.float64,
    )


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0) if x.size else np.zeros((x.shape[1],), dtype=np.float64)
    scale = x.std(axis=0) if x.size else np.ones((x.shape[1],), dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    return (x - mean) / scale, mean, scale


def _fit_ridge(x: np.ndarray, y: np.ndarray, *, alpha: float) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros((x.shape[1] + 1,), dtype=np.float64)
    xz, _mean, _scale = _standardize(x)
    design = np.column_stack([xz, np.ones(xz.shape[0], dtype=np.float64)])
    penalty = float(alpha) * np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    return np.linalg.pinv(design.T @ design + penalty) @ design.T @ y


def _predict_with_training_stats(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    alpha: float,
) -> np.ndarray:
    xz_train, mean, scale = _standardize(x_train)
    design_train = np.column_stack([xz_train, np.ones(xz_train.shape[0], dtype=np.float64)])
    penalty = float(alpha) * np.eye(design_train.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    weights = np.linalg.pinv(design_train.T @ design_train + penalty) @ design_train.T @ y_train
    xz_eval = (x_eval - mean) / scale
    design_eval = np.column_stack([xz_eval, np.ones(xz_eval.shape[0], dtype=np.float64)])
    return np.asarray(design_eval @ weights, dtype=np.float64)


def leave_one_out_predictions(rows: list[dict[str, str]], *, alpha: float) -> list[float]:
    x = _feature_matrix(rows)
    y = np.asarray([_target(row) for row in rows], dtype=np.float64)
    predictions: list[float] = []
    for index in range(len(rows)):
        train_mask = np.ones(len(rows), dtype=bool)
        train_mask[index] = False
        if int(np.sum(train_mask)) == 0:
            predictions.append(0.0)
            continue
        pred = _predict_with_training_stats(
            x_train=x[train_mask],
            y_train=y[train_mask],
            x_eval=x[index : index + 1],
            alpha=alpha,
        )[0]
        predictions.append(float(pred))
    return predictions


def build_acceptor_report(
    *,
    calibration_rows: list[dict[str, str]],
    alpha: float = 5.0,
    uncertainty_scale: float = 1.0,
    min_outcome_count: int = 5,
) -> dict[str, Any]:
    if not calibration_rows:
        return {"rows": [], "recommended": [], "residual_std": 0.0, "feature_columns": list(FEATURE_COLUMNS)}
    x = _feature_matrix(calibration_rows)
    y = np.asarray([_target(row) for row in calibration_rows], dtype=np.float64)
    loo = leave_one_out_predictions(calibration_rows, alpha=alpha)
    residuals = np.asarray([target - pred for target, pred in zip(y, loo, strict=True)], dtype=np.float64)
    residual_std = float(np.std(residuals)) if len(residuals) > 1 else 0.0
    full_pred = _predict_with_training_stats(x_train=x, y_train=y, x_eval=x, alpha=alpha)
    rows: list[dict[str, Any]] = []
    for row, observed, loo_pred, pred in zip(calibration_rows, y, loo, full_pred, strict=True):
        conservative_score = float(pred) - float(uncertainty_scale) * residual_std
        outcome_count = int(round(_safe_float(row.get("outcome_count"))))
        enough_outcomes = int(outcome_count >= int(min_outcome_count))
        observed_gate = int(
            enough_outcomes > 0
            and
            _safe_float(row.get("clears_stats_gate")) > 0.0
            and _safe_float(row.get("clears_best_random_gate")) > 0.0
        )
        rows.append(
            {
                "label": row.get("label", ""),
                "changed_count": int(round(_safe_float(row.get("changed_count")))),
                "observed_target": float(observed),
                "loo_prediction": float(loo_pred),
                "prediction": float(pred),
                "conservative_score": conservative_score,
                "enough_outcomes": enough_outcomes,
                "observed_gate": observed_gate,
                "recommend": int(conservative_score > 0.0 and observed_gate > 0),
                "mean_delta_to_stats": _safe_float(row.get("mean_delta_to_stats")),
                "mean_delta_to_best_random": _safe_float(row.get("mean_delta_to_best_random")),
                "outcome_count": outcome_count,
            }
        )
    rows.sort(key=lambda item: (item["conservative_score"], item["observed_target"]), reverse=True)
    recommended = [row for row in rows if row["recommend"]]
    return {
        "alpha": float(alpha),
        "uncertainty_scale": float(uncertainty_scale),
        "min_outcome_count": int(min_outcome_count),
        "residual_std": residual_std,
        "feature_columns": list(FEATURE_COLUMNS),
        "rows": rows,
        "recommended": recommended,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="outputs/dfr_sweeps/waterbirds-replacement-calibration.csv")
    parser.add_argument("--output-csv", default="outputs/dfr_sweeps/waterbirds-replacement-acceptor.csv")
    parser.add_argument("--output-json", default="outputs/dfr_sweeps/waterbirds-replacement-acceptor.json")
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--uncertainty-scale", type=float, default=1.0)
    parser.add_argument("--min-outcome-count", type=int, default=5)
    args = parser.parse_args()
    rows = _read_rows(Path(args.input_csv))
    report = build_acceptor_report(
        calibration_rows=rows,
        alpha=float(args.alpha),
        uncertainty_scale=float(args.uncertainty_scale),
        min_outcome_count=int(args.min_outcome_count),
    )
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    write_csv(output_csv, report["rows"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_csv": str(output_csv),
                "output_json": str(output_json),
                "recommended": [row["label"] for row in report["recommended"]],
                "residual_std": report["residual_std"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
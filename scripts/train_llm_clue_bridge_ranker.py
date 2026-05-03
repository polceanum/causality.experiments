from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FEATURE_COLUMNS = [
    "label_corr",
    "env_corr",
    "corr_margin",
    "abs_corr_margin",
    "uncertainty",
    "top_group_entropy",
    "label_env_disentanglement",
]


class BridgeRankerModel(NamedTuple):
    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    train_trace_count: int
    run_count: int


class PairwiseBridgeRankerModel(NamedTuple):
    weights: np.ndarray
    scale: np.ndarray
    train_pair_count: int
    train_trace_count: int
    run_count: int


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _dataset_name(run_dir: Path) -> str:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = str(manifest.get("config", ""))
        if config:
            return Path(config).stem
        dataset = str(manifest.get("dataset", ""))
        if dataset:
            return dataset
    return run_dir.name


def _feature_vector(row: dict[str, Any]) -> list[float]:
    values = [_safe_float(row.get(column)) for column in FEATURE_COLUMNS]
    values.append(_safe_float(row.get("label_corr")) - _safe_float(row.get("env_corr")))
    values.append(1.0)
    return values


def _trace_target(row: dict[str, Any]) -> float:
    test_value = max(_safe_float(row.get("test_value")), 0.0)
    score_delta = max(_safe_float(row.get("score_delta")), 0.0)
    correct_bonus = 0.25 if bool(row.get("hypothesis_correct", False)) else 0.0
    passed_bonus = 0.25 if bool(row.get("passed_control", False)) else 0.0
    return test_value + score_delta + correct_bonus + passed_bonus


def _fit_ridge(rows: list[dict[str, Any]], *, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("No bridge training rows found.")
    x = np.asarray([_feature_vector(row) for row in rows], dtype=np.float64)
    y = np.asarray([_trace_target(row) for row in rows], dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    xz = (x - mean) / scale
    penalty = alpha * np.eye(xz.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    weights = np.linalg.pinv(xz.T @ xz + penalty) @ xz.T @ y
    return weights, mean, scale


def _fit_pairwise_ridge(
    rows: list[dict[str, Any]],
    *,
    alpha: float,
    min_target_delta: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not rows:
        raise ValueError("No bridge training rows found.")
    features = [np.asarray(_feature_vector(row), dtype=np.float64) for row in rows]
    targets = [_trace_target(row) for row in rows]
    pair_features: list[np.ndarray] = []
    pair_targets: list[float] = []
    for left_index, right_index in itertools.combinations(range(len(rows)), 2):
        delta = targets[left_index] - targets[right_index]
        if abs(delta) <= min_target_delta:
            continue
        diff = features[left_index] - features[right_index]
        pair_features.append(diff)
        pair_targets.append(delta)
        pair_features.append(-diff)
        pair_targets.append(-delta)
    if not pair_features:
        raise ValueError("No informative bridge training pairs found.")
    x = np.asarray(pair_features, dtype=np.float64)
    y = np.asarray(pair_targets, dtype=np.float64)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    xz = x / scale
    penalty = alpha * np.eye(xz.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    weights = np.linalg.pinv(xz.T @ xz + penalty) @ xz.T @ y
    return weights, scale, len(pair_targets)


def _predict(row: dict[str, Any], weights: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> float:
    x = np.asarray(_feature_vector(row), dtype=np.float64)
    return float(((x - mean) / scale) @ weights)


def _predict_pairwise(row: dict[str, Any], weights: np.ndarray, scale: np.ndarray) -> float:
    x = np.asarray(_feature_vector(row), dtype=np.float64)
    return float((x / scale) @ weights)


def _matches_any_dataset_name(dataset: str, patterns: list[str] | None) -> bool:
    if not patterns:
        return False
    lowered = dataset.lower()
    return any(pattern.strip().lower() in lowered for pattern in patterns if pattern.strip())


def fit_bridge_ranker_from_runs(
    input_dir: Path,
    *,
    alpha: float = 10.0,
    exclude_datasets: list[str] | None = None,
) -> BridgeRankerModel:
    run_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not run_dirs:
        raise ValueError(f"No run directories found under {input_dir}.")
    train_traces: list[dict[str, Any]] = []
    included_runs = 0
    for run_dir in run_dirs:
        dataset = _dataset_name(run_dir)
        if _matches_any_dataset_name(dataset, exclude_datasets):
            continue
        traces = _read_jsonl(run_dir / "training_traces.jsonl")
        if traces:
            train_traces.extend(traces)
            included_runs += 1
    weights, mean, scale = _fit_ridge(train_traces, alpha=alpha)
    return BridgeRankerModel(
        weights=weights,
        mean=mean,
        scale=scale,
        train_trace_count=len(train_traces),
        run_count=included_runs,
    )


def fit_pairwise_bridge_ranker_from_runs(
    input_dir: Path,
    *,
    alpha: float = 10.0,
    exclude_datasets: list[str] | None = None,
) -> PairwiseBridgeRankerModel:
    run_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not run_dirs:
        raise ValueError(f"No run directories found under {input_dir}.")
    train_traces: list[dict[str, Any]] = []
    included_runs = 0
    for run_dir in run_dirs:
        dataset = _dataset_name(run_dir)
        if _matches_any_dataset_name(dataset, exclude_datasets):
            continue
        traces = _read_jsonl(run_dir / "training_traces.jsonl")
        if traces:
            train_traces.extend(traces)
            included_runs += 1
    weights, scale, pair_count = _fit_pairwise_ridge(train_traces, alpha=alpha)
    return PairwiseBridgeRankerModel(
        weights=weights,
        scale=scale,
        train_pair_count=pair_count,
        train_trace_count=len(train_traces),
        run_count=included_runs,
    )


def score_bridge_packets(packets: list[dict[str, Any]], model: BridgeRankerModel) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for packet in packets:
        score = _predict(packet, model.weights, model.mean, model.scale)
        rows.append(
            {
                "dataset": str(packet.get("dataset", "")),
                "feature_index": str(packet.get("feature_index", "")),
                "feature_name": str(packet.get("feature_name", "")),
                "support_score": f"{score:.6f}",
                "rank_score": f"{score:.6f}",
                "score": f"{score:.6f}",
                "score_source": "bridge",
            }
        )
    return rows


def score_pairwise_bridge_packets(packets: list[dict[str, Any]], model: PairwiseBridgeRankerModel) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for packet in packets:
        score = _predict_pairwise(packet, model.weights, model.scale)
        rows.append(
            {
                "dataset": str(packet.get("dataset", "")),
                "feature_index": str(packet.get("feature_index", "")),
                "feature_name": str(packet.get("feature_name", "")),
                "support_score": f"{score:.6f}",
                "rank_score": f"{score:.6f}",
                "score": f"{score:.6f}",
                "score_source": "pairwise_bridge",
            }
        )
    return rows


def _random_score(feature_index: int) -> float:
    return ((feature_index * 1103515245 + 12345) % 1000000) / 1000000.0


def _minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if high - low < 1e-12:
        return {key: 0.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def _fuse_scores(
    *,
    ranker_scores: dict[str, float],
    stats_scores: dict[str, float],
    weight: float,
) -> dict[str, float]:
    normalized_ranker = _minmax_normalize(ranker_scores)
    normalized_stats = _minmax_normalize(stats_scores)
    clipped_weight = min(max(float(weight), 0.0), 1.0)
    return {
        name: clipped_weight * normalized_ranker.get(name, 0.0) + (1.0 - clipped_weight) * normalized_stats.get(name, 0.0)
        for name in stats_scores
    }


def _evaluate_scores(
    *,
    dataset: str,
    packets: list[dict[str, Any]],
    feature_clues: dict[str, dict[str, str]],
    scores: dict[str, dict[str, float]],
    top_k_values: list[int],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label, by_feature in scores.items():
        ranked = sorted(packets, key=lambda packet: by_feature.get(str(packet.get("feature_name", "")), 0.0), reverse=True)
        for top_k in top_k_values:
            selected = ranked[: min(top_k, len(ranked))]
            if not selected:
                continue
            causal_values = []
            selected_names = []
            for packet in selected:
                name = str(packet.get("feature_name", ""))
                selected_names.append(name)
                clue = feature_clues.get(name, {})
                causal_values.append(_safe_float(clue.get("causal_target"), _safe_float(packet.get("causal_target"))))
            rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "top_k": str(top_k),
                    "selected_count": str(len(selected)),
                    "mean_causal_target": f"{statistics.mean(causal_values):.6f}",
                    "selected_features": ";".join(selected_names),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_bridge_ranker(
    *,
    input_dir: Path,
    output_csv: Path,
    output_json: Path,
    alpha: float = 10.0,
    top_k_values: list[int] | None = None,
    fusion_weights: list[float] | None = None,
) -> dict[str, Any]:
    run_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not run_dirs:
        raise ValueError(f"No run directories found under {input_dir}.")
    top_k_values = top_k_values or [1, 2, 4]
    fusion_weights = fusion_weights or [0.1, 0.3, 0.5]
    all_rows: list[dict[str, str]] = []
    dataset_summaries: list[dict[str, Any]] = []

    for heldout in run_dirs:
        train_traces: list[dict[str, Any]] = []
        for run_dir in run_dirs:
            if run_dir == heldout:
                continue
            train_traces.extend(_read_jsonl(run_dir / "training_traces.jsonl"))
        weights, mean, scale = _fit_ridge(train_traces, alpha=alpha)
        pairwise_weights, pairwise_scale, train_pair_count = _fit_pairwise_ridge(train_traces, alpha=alpha)

        packets = _read_jsonl(heldout / "latent_clue_packets.jsonl")
        if not packets:
            continue
        feature_clues = {str(row.get("feature_name", "")): row for row in _read_csv(heldout / "feature_clues.csv")}
        dataset = _dataset_name(heldout)

        bridge_scores: dict[str, float] = {}
        pairwise_scores: dict[str, float] = {}
        stats_scores: dict[str, float] = {}
        random_scores: dict[str, float] = {}
        for packet in packets:
            name = str(packet.get("feature_name", ""))
            feature_index = int(_safe_float(packet.get("feature_index")))
            bridge_scores[name] = _predict(packet, weights, mean, scale)
            pairwise_scores[name] = _predict_pairwise(packet, pairwise_weights, pairwise_scale)
            stats_scores[name] = _safe_float(packet.get("label_corr")) - _safe_float(packet.get("env_corr"))
            random_scores[name] = _random_score(feature_index)

        rows = _evaluate_scores(
            dataset=dataset,
            packets=packets,
            feature_clues=feature_clues,
            scores={
                "bridge_ranker": bridge_scores,
                "pairwise_bridge_ranker": pairwise_scores,
                **{
                    f"pairwise_stats_fused_w{weight:g}": _fuse_scores(
                        ranker_scores=pairwise_scores,
                        stats_scores=stats_scores,
                        weight=weight,
                    )
                    for weight in fusion_weights
                },
                "stats_margin": stats_scores,
                "random": random_scores,
            },
            top_k_values=top_k_values,
        )
        all_rows.extend(rows)
        dataset_summaries.append(
            {
                "dataset": dataset,
                "train_trace_count": len(train_traces),
                "train_pair_count": train_pair_count,
                "packet_count": len(packets),
            }
        )

    _write_csv(output_csv, all_rows)
    aggregate: dict[str, Any] = {
        "input_dir": str(input_dir),
        "alpha": float(alpha),
        "top_k_values": top_k_values,
        "fusion_weights": fusion_weights,
        "datasets": dataset_summaries,
        "by_label_top_k": [],
    }
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in all_rows:
        grouped.setdefault((row["label"], row["top_k"]), []).append(float(row["mean_causal_target"]))
    for (label, top_k), values in sorted(grouped.items()):
        aggregate["by_label_top_k"].append(
            {
                "label": label,
                "top_k": int(top_k),
                "count": len(values),
                "mean_causal_target": statistics.mean(values),
                "std_causal_target": statistics.pstdev(values) if len(values) > 1 else 0.0,
            }
        )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/dfr_sweeps/llm_clue_fixture_experiments")
    parser.add_argument("--output-csv", default="outputs/dfr_sweeps/llm_clue_bridge_ranker_heldout.csv")
    parser.add_argument("--output-json", default="outputs/dfr_sweeps/llm_clue_bridge_ranker_heldout.json")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--top-k", nargs="*", type=int, default=[1, 2, 4])
    parser.add_argument("--fusion-weights", nargs="*", type=float, default=[0.1, 0.3, 0.5])
    args = parser.parse_args()
    summary = run_bridge_ranker(
        input_dir=Path(args.input_dir),
        output_csv=Path(args.output_csv),
        output_json=Path(args.output_json),
        alpha=float(args.alpha),
        top_k_values=list(args.top_k),
        fusion_weights=list(args.fusion_weights),
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

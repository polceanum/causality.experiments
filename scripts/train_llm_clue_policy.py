from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.rl_clue_policy import (
    build_clue_reward_rows,
    score_policy_packets,
    train_offline_clue_policy,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _reward_rows_for_run(run_dir: Path) -> list[dict[str, Any]]:
    packets = _read_jsonl(run_dir / "latent_clue_packets.jsonl")
    traces = _read_jsonl(run_dir / "training_traces.jsonl")
    feature_clues = {str(row.get("feature_name", "")): row for row in _read_csv(run_dir / "feature_clues.csv")}
    return build_clue_reward_rows(
        packets=packets,
        traces=traces,
        feature_clues=feature_clues,
        dataset=_dataset_name(run_dir),
        reward_scope="fixture",
    )


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


def run_clue_policy_training(
    *,
    input_dir: Path,
    reward_csv: Path,
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
    all_reward_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        all_reward_rows.extend(_reward_rows_for_run(run_dir))
    _write_csv(reward_csv, all_reward_rows)

    all_eval_rows: list[dict[str, str]] = []
    dataset_summaries: list[dict[str, Any]] = []
    for heldout in run_dirs:
        train_rows: list[dict[str, Any]] = []
        for run_dir in run_dirs:
            if run_dir != heldout:
                train_rows.extend(_reward_rows_for_run(run_dir))
        policy = train_offline_clue_policy(train_rows, alpha=alpha)
        packets = _read_jsonl(heldout / "latent_clue_packets.jsonl")
        if not packets:
            continue
        feature_clues = {str(row.get("feature_name", "")): row for row in _read_csv(heldout / "feature_clues.csv")}
        policy_scores = {row["feature_name"]: _safe_float(row.get("score")) for row in score_policy_packets(packets, policy)}
        stats_scores: dict[str, float] = {}
        random_scores: dict[str, float] = {}
        for packet in packets:
            name = str(packet.get("feature_name", ""))
            stats_scores[name] = _safe_float(packet.get("label_corr")) - _safe_float(packet.get("env_corr"))
            random_scores[name] = _random_score(int(_safe_float(packet.get("feature_index"))))
        normalized_policy = _minmax_normalize(policy_scores)
        normalized_stats = _minmax_normalize(stats_scores)
        score_sets = {
            "offline_clue_policy": policy_scores,
            "stats_margin": stats_scores,
            "random": random_scores,
        }
        for weight in fusion_weights:
            label = f"policy_stats_fused_w{weight:g}"
            score_sets[label] = {
                name: float(weight) * normalized_policy.get(name, 0.0)
                + (1.0 - float(weight)) * normalized_stats.get(name, 0.0)
                for name in stats_scores
            }
        dataset = _dataset_name(heldout)
        all_eval_rows.extend(
            _evaluate_scores(
                dataset=dataset,
                packets=packets,
                feature_clues=feature_clues,
                scores=score_sets,
                top_k_values=top_k_values,
            )
        )
        dataset_summaries.append(
            {
                "dataset": dataset,
                "train_reward_count": policy.train_reward_count,
                "packet_count": len(packets),
            }
        )

    _write_csv(output_csv, all_eval_rows)
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in all_eval_rows:
        grouped.setdefault((row["label"], row["top_k"]), []).append(float(row["mean_causal_target"]))
    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "reward_csv": str(reward_csv),
        "alpha": float(alpha),
        "top_k_values": top_k_values,
        "fusion_weights": fusion_weights,
        "datasets": dataset_summaries,
        "by_label_top_k": [],
    }
    for (label, top_k), values in sorted(grouped.items()):
        summary["by_label_top_k"].append(
            {
                "label": label,
                "top_k": int(top_k),
                "count": len(values),
                "mean_causal_target": statistics.mean(values),
                "std_causal_target": statistics.pstdev(values) if len(values) > 1 else 0.0,
            }
        )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/dfr_sweeps/llm_clue_fixture_experiments_20260502_refreshed")
    parser.add_argument("--reward-csv", default="outputs/dfr_sweeps/llm_clue_policy_rewards.csv")
    parser.add_argument("--output-csv", default="outputs/dfr_sweeps/llm_clue_policy_heldout.csv")
    parser.add_argument("--output-json", default="outputs/dfr_sweeps/llm_clue_policy_heldout.json")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--top-k", nargs="*", type=int, default=[1, 2, 4])
    parser.add_argument("--fusion-weights", nargs="*", type=float, default=[0.1, 0.3, 0.5])
    args = parser.parse_args()
    summary = run_clue_policy_training(
        input_dir=Path(args.input_dir),
        reward_csv=Path(args.reward_csv),
        output_csv=Path(args.output_csv),
        output_json=Path(args.output_json),
        alpha=float(args.alpha),
        top_k_values=list(args.top_k),
        fusion_weights=list(args.fusion_weights),
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
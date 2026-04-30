from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.clues import build_feature_cards, build_language_clue_rows, write_csv_rows
from causality_experiments.config import load_config
from causality_experiments.data import load_dataset
from causality_experiments.discovery import build_feature_clue_rows, merge_external_clue_rows
from causality_experiments.run import run_experiment
from scripts.report_clue_source_ablation import summarize_source_ablation


SOURCE_LABELS = ("stats", "language", "fused")


def _safe_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def source_score(row: dict[str, Any], source: str) -> float:
    stats_score = _safe_float(row, "soft_causal_target")
    if stats_score <= 0.0:
        stats_score = _sigmoid(6.0 * _safe_float(row, "corr_margin"))
    language_confidence = min(max(_safe_float(row, "language_confidence"), 0.0), 1.0)
    language_causal = min(max(_safe_float(row, "language_causal_score"), 0.0), 1.0)
    language_spurious = min(max(_safe_float(row, "language_spurious_score"), 0.0), 1.0)
    language_score = min(max(language_causal * language_confidence + 0.5 * (1.0 - language_confidence), 0.0), 1.0)
    language_penalized = min(max(language_score - 0.25 * language_confidence * language_spurious, 0.0), 1.0)
    if source == "stats":
        return stats_score
    if source == "language":
        return language_penalized
    if source == "fused":
        language_weight = 0.5 * language_confidence
        return (1.0 - language_weight) * stats_score + language_weight * language_penalized
    raise ValueError(f"Unknown clue score source {source!r}.")


def build_source_score_rows(clue_rows: list[dict[str, Any]], source: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in clue_rows:
        score = source_score(row, source)
        rows.append(
            {
                "dataset": str(row.get("dataset", "")),
                "feature_index": str(row.get("feature_index", "")),
                "feature_name": str(row.get("feature_name", "")),
                "support_score": f"{score:.6f}",
                "rank_score": f"{score:.6f}",
                "score": f"{score:.6f}",
                "score_source": source,
            }
        )
    return rows


def resolve_sources(values: list[str] | None) -> list[str]:
    if not values:
        return list(SOURCE_LABELS)
    sources: list[str] = []
    for value in values:
        for part in value.split(","):
            source = part.strip().lower()
            if not source:
                continue
            if source not in SOURCE_LABELS:
                raise ValueError(f"Unknown source {source!r}. Known sources: {', '.join(SOURCE_LABELS)}")
            if source not in sources:
                sources.append(source)
    if not sources:
        raise ValueError("At least one source must be selected.")
    return sources


def build_downstream_candidate(
    base_config: dict[str, Any],
    *,
    label: str,
    top_k: int,
    score_path: Path | None = None,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    base_name = str(config.get("name", "waterbirds_features"))
    config["name"] = f"{base_name}_clue_{label}_top{top_k}"
    dataset = dict(config.get("dataset", {}))
    if label in SOURCE_LABELS:
        if score_path is None:
            raise ValueError(f"{label} candidate requires a score path.")
        dataset["causal_mask_strategy"] = "discovery_scores"
        dataset["discovery_scores_path"] = str(score_path)
        dataset["discovery_score_threshold"] = 2.0
        dataset["discovery_score_top_k"] = int(top_k)
    elif label == "heuristic":
        dataset["causal_mask_strategy"] = str(dataset.get("causal_mask_strategy", "label_minus_env_correlation"))
        dataset["causal_mask_top_k"] = int(top_k)
        dataset.pop("discovery_scores_path", None)
        dataset.pop("discovery_score_threshold", None)
        dataset.pop("discovery_score_top_k", None)
    elif label == "random":
        dataset["causal_mask_strategy"] = "random_top_k"
        dataset["causal_mask_top_k"] = int(top_k)
        dataset["causal_mask_random_seed"] = int(config.get("seed", 0))
        dataset.pop("discovery_scores_path", None)
        dataset.pop("discovery_score_threshold", None)
        dataset.pop("discovery_score_top_k", None)
    else:
        raise ValueError(f"Unknown downstream candidate label {label!r}.")
    config["dataset"] = dataset
    return config


def with_dataset_path(config: dict[str, Any], dataset_path: str | None) -> dict[str, Any]:
    if not dataset_path:
        return config
    updated = deepcopy(config)
    dataset = dict(updated.get("dataset", {}))
    dataset["path"] = dataset_path
    updated["dataset"] = dataset
    return updated


def with_runtime_overrides(
    config: dict[str, Any],
    *,
    official_dfr_num_retrains: int | None = None,
    training_device: str | None = None,
) -> dict[str, Any]:
    updated = deepcopy(config)
    if official_dfr_num_retrains is not None:
        method = dict(updated.get("method", {}))
        method["official_dfr_num_retrains"] = int(official_dfr_num_retrains)
        updated["method"] = method
    if training_device:
        training = dict(updated.get("training", {}))
        training["device"] = training_device
        updated["training"] = training
    return updated


def run_downstream_candidates(
    base_config: dict[str, Any],
    score_paths: dict[str, Path],
    *,
    top_k_values: list[int],
    output_root: Path,
    include_heuristic: bool,
    include_random: bool,
) -> list[dict[str, Any]]:
    labels = list(score_paths)
    if include_heuristic:
        labels.append("heuristic")
    if include_random:
        labels.append("random")
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        for label in labels:
            for top_k in top_k_values:
                candidate = build_downstream_candidate(base_config, label=label, top_k=top_k, score_path=score_paths.get(label))
                config_path = tmp_root / f"{candidate['name']}.yaml"
                config_path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
                run_dir = run_experiment(config_path, output_root)
                payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                metrics = payload["metrics"]
                rows.append(
                    {
                        "label": label,
                        "top_k": top_k,
                        "config": candidate["name"],
                        "run": run_dir.name,
                        "val_wga": metrics.get("val/worst_group_accuracy", float("nan")),
                        "test_wga": metrics.get("test/worst_group_accuracy", float("nan")),
                        "val_acc": metrics.get("val/accuracy", float("nan")),
                        "test_acc": metrics.get("test/accuracy", float("nan")),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True, help="Waterbirds feature config used for cards, clues, and optional downstream runs.")
    parser.add_argument("--dataset-path", default="", help="Optional feature CSV path override for the base config dataset.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--top-k", type=int, action="append", default=[], help="Top-k to summarize/run. Can be passed multiple times.")
    parser.add_argument("--sources", action="append", default=[], help="Source label(s) to emit/run: stats, language, fused. Can be comma-separated or repeated.")
    parser.add_argument("--out-dir", default="outputs/dfr_sweeps/clue_fusion", help="Directory for cards, clues, scores, ablations, and optional downstream rows.")
    parser.add_argument("--card-top-k", type=int, default=16, help="Top and bottom activation count per feature card.")
    parser.add_argument("--run-downstream", action="store_true", help="Run downstream top-k DFR candidates after writing source score files.")
    parser.add_argument("--include-heuristic", action="store_true")
    parser.add_argument("--include-random", action="store_true")
    parser.add_argument("--official-dfr-num-retrains", type=int, default=None, help="Optional compact-screen override for official DFR retrain count.")
    parser.add_argument("--training-device", default="", help="Optional training.device override for downstream runs.")
    parser.add_argument("--output-root", default="outputs/runs")
    args = parser.parse_args()

    top_k_values = args.top_k or [64, 128, 256]
    sources = resolve_sources(args.sources)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_config = with_runtime_overrides(
        with_dataset_path(load_config(Path(args.base_config)), args.dataset_path or None),
        official_dfr_num_retrains=args.official_dfr_num_retrains,
        training_device=args.training_device or None,
    )
    bundle = load_dataset(base_config)

    cards = build_feature_cards(bundle, split_name=args.split, top_k=args.card_top_k)
    language_clues = build_language_clue_rows(cards, domain="waterbirds")
    clue_rows = merge_external_clue_rows(build_feature_clue_rows(bundle, split_name=args.split), language_clues)

    cards_path = out_dir / "feature_cards.csv"
    language_path = out_dir / "language_clues.csv"
    clue_path = out_dir / "merged_clues.csv"
    write_csv_rows(cards_path, cards)
    write_csv_rows(language_path, language_clues)
    write_csv_rows(clue_path, clue_rows)

    score_paths: dict[str, Path] = {}
    for source in sources:
        score_path = out_dir / f"scores_{source}.csv"
        write_csv_rows(score_path, build_source_score_rows(clue_rows, source))
        score_paths[source] = score_path

    ablation_rows = summarize_source_ablation(
        clue_path,
        [(label, path) for label, path in score_paths.items()],
        top_k_values=top_k_values,
        reference_label="stats",
    )
    ablation_path = out_dir / "source_ablation.csv"
    write_csv_rows(ablation_path, ablation_rows)

    manifest: dict[str, Any] = {
        "base_config": str(args.base_config),
        "dataset_path": str(base_config.get("dataset", {}).get("path", "")),
        "split": args.split,
        "top_k": top_k_values,
        "sources": sources,
        "official_dfr_num_retrains": args.official_dfr_num_retrains,
        "training_device": args.training_device,
        "cards": str(cards_path),
        "language_clues": str(language_path),
        "merged_clues": str(clue_path),
        "scores": {label: str(path) for label, path in score_paths.items()},
        "source_ablation": str(ablation_path),
    }
    if args.run_downstream:
        downstream_rows = run_downstream_candidates(
            base_config,
            score_paths,
            top_k_values=top_k_values,
            output_root=Path(args.output_root),
            include_heuristic=args.include_heuristic,
            include_random=args.include_random,
        )
        downstream_path = out_dir / "downstream_results.csv"
        write_csv_rows(downstream_path, downstream_rows)
        manifest["downstream_results"] = str(downstream_path)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()

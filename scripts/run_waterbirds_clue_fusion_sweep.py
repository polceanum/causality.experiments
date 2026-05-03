from __future__ import annotations

import argparse
import csv
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

from causality_experiments.clues import (
    build_feature_cards,
    build_image_prototype_clue_rows,
    build_language_clue_rows,
    write_csv_rows,
)
from causality_experiments.config import load_config
from causality_experiments.data import load_dataset
from causality_experiments.discovery import build_feature_clue_rows, merge_external_clue_rows
from causality_experiments.latent_clue_packets import build_latent_clue_packets
from causality_experiments.rl_clue_policy import build_clue_reward_rows, score_policy_packets, train_offline_clue_policy
from causality_experiments.run import run_experiment
from scripts.report_clue_source_ablation import summarize_source_ablation
from scripts.train_llm_clue_bridge_ranker import (
    fit_bridge_ranker_from_runs,
    fit_pairwise_bridge_ranker_from_runs,
    score_bridge_packets,
    score_pairwise_bridge_packets,
)


DEFAULT_SOURCE_LABELS = ("stats", "language", "image", "fused")
SOURCE_LABELS = (
    *DEFAULT_SOURCE_LABELS,
    "bridge",
    "bridge_fused",
    "bridge_gated",
    "pairwise_bridge",
    "pairwise_bridge_fused",
    "policy",
    "policy_fused",
    "policy_safe",
)


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


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = value.strip().lower()
        if key and key not in seen:
            output.append(value)
            seen.add(key)
    return output


def _normalise_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo + 1e-12:
        return [0.5 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def _matches_any_dataset_name(dataset: str, patterns: list[str] | None) -> bool:
    if not patterns:
        return False
    lowered = dataset.lower()
    return any(pattern.strip().lower() in lowered for pattern in patterns if pattern.strip())


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


def source_score(row: dict[str, Any], source: str) -> float:
    stats_score = _safe_float(row, "soft_causal_target")
    if stats_score <= 0.0:
        stats_score = _sigmoid(6.0 * _safe_float(row, "corr_margin"))
    language_confidence = min(max(_safe_float(row, "language_confidence"), 0.0), 1.0)
    language_causal = min(max(_safe_float(row, "language_causal_score"), 0.0), 1.0)
    language_spurious = min(max(_safe_float(row, "language_spurious_score"), 0.0), 1.0)
    language_score = min(max(language_causal * language_confidence + 0.5 * (1.0 - language_confidence), 0.0), 1.0)
    language_penalized = min(max(language_score - 0.25 * language_confidence * language_spurious, 0.0), 1.0)
    image_confidence = min(max(_safe_float(row, "image_confidence"), 0.0), 1.0)
    image_label = min(max(_safe_float(row, "image_label_score"), 0.0), 1.0)
    image_background = min(max(_safe_float(row, "image_background_score"), 0.0), 1.0)
    image_score = min(max(image_label * image_confidence + 0.5 * (1.0 - image_confidence), 0.0), 1.0)
    image_penalized = min(max(image_score - 0.25 * image_confidence * image_background, 0.0), 1.0)
    if source == "stats":
        return stats_score
    if source == "language":
        return language_penalized
    if source == "image":
        return image_penalized
    if source == "fused":
        language_weight = 0.35 * language_confidence
        image_weight = 0.35 * image_confidence
        evidence_weight = min(language_weight + image_weight, 0.7)
        if evidence_weight <= 0.0:
            return stats_score
        clue_score = (language_weight * language_penalized + image_weight * image_penalized) / max(
            language_weight + image_weight,
            1e-12,
        )
        return (1.0 - evidence_weight) * stats_score + evidence_weight * clue_score
    if source == "bridge":
        return _safe_float(row, "bridge_score")
    if source == "bridge_fused":
        return _safe_float(row, "bridge_fused_score")
    if source == "bridge_gated":
        return _safe_float(row, "bridge_gated_score")
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


def build_bridge_score_rows(
    bundle: Any,
    *,
    bridge_input_dir: Path,
    alpha: float = 10.0,
    exclude_datasets: list[str] | None = None,
    split_name: str = "train",
    card_top_k: int = 16,
    blend_with_stats_weight: float | None = None,
    blend_mode: str = "linear",
) -> list[dict[str, str]]:
    packets = build_latent_clue_packets(bundle, split_name=split_name, top_k=card_top_k)
    model = fit_bridge_ranker_from_runs(
        bridge_input_dir,
        alpha=alpha,
        exclude_datasets=exclude_datasets,
    )
    rows = score_bridge_packets(packets, model)
    if blend_with_stats_weight is None:
        return rows
    bridge_values = _normalise_scores([_safe_float(row, "score") for row in rows])
    weight = min(max(float(blend_with_stats_weight), 0.0), 1.0)
    mode = blend_mode.strip().lower()
    blended_rows: list[dict[str, str]] = []
    for row, packet, bridge_value in zip(rows, packets, bridge_values, strict=True):
        stats_score = _sigmoid(6.0 * _safe_float(packet, "corr_margin"))
        if mode in {"linear", "fused", "blend"}:
            score = (1.0 - weight) * stats_score + weight * bridge_value
            score_source = "bridge_fused"
        elif mode in {"gated", "multiplicative"}:
            score = stats_score * (1.0 + weight * bridge_value)
            score_source = "bridge_gated"
        else:
            raise ValueError("blend_mode must be 'linear' or 'gated'.")
        blended = dict(row)
        blended["support_score"] = f"{score:.6f}"
        blended["rank_score"] = f"{score:.6f}"
        blended["score"] = f"{score:.6f}"
        blended["score_source"] = score_source
        blended_rows.append(blended)
    return blended_rows


def build_pairwise_bridge_score_rows(
    bundle: Any,
    *,
    bridge_input_dir: Path,
    alpha: float = 10.0,
    exclude_datasets: list[str] | None = None,
    split_name: str = "train",
    card_top_k: int = 16,
    blend_with_stats_weight: float | None = None,
) -> list[dict[str, str]]:
    packets = build_latent_clue_packets(bundle, split_name=split_name, top_k=card_top_k)
    model = fit_pairwise_bridge_ranker_from_runs(
        bridge_input_dir,
        alpha=alpha,
        exclude_datasets=exclude_datasets,
    )
    rows = score_pairwise_bridge_packets(packets, model)
    if blend_with_stats_weight is None:
        return [dict(row, score_source="pairwise_bridge") for row in rows]
    pairwise_values = _normalise_scores([_safe_float(row, "score") for row in rows])
    weight = min(max(float(blend_with_stats_weight), 0.0), 1.0)
    blended_rows: list[dict[str, str]] = []
    for row, packet, pairwise_value in zip(rows, packets, pairwise_values, strict=True):
        stats_score = _sigmoid(6.0 * _safe_float(packet, "corr_margin"))
        score = (1.0 - weight) * stats_score + weight * pairwise_value
        blended = dict(row)
        blended["support_score"] = f"{score:.6f}"
        blended["rank_score"] = f"{score:.6f}"
        blended["score"] = f"{score:.6f}"
        blended["score_source"] = "pairwise_bridge_fused"
        blended_rows.append(blended)
    return blended_rows


def build_policy_score_rows(
    bundle: Any,
    *,
    policy_input_dir: Path,
    alpha: float = 10.0,
    exclude_datasets: list[str] | None = None,
    split_name: str = "train",
    card_top_k: int = 16,
    blend_with_stats_weight: float | None = None,
    blend_mode: str = "safe_residual",
) -> list[dict[str, str]]:
    train_rows: list[dict[str, Any]] = []
    run_dirs = sorted(path for path in policy_input_dir.iterdir() if path.is_dir())
    if not run_dirs:
        raise ValueError(f"No run directories found under {policy_input_dir}.")
    for run_dir in run_dirs:
        dataset = _dataset_name(run_dir)
        if _matches_any_dataset_name(dataset, exclude_datasets):
            continue
        train_rows.extend(_reward_rows_for_run(run_dir))
    policy = train_offline_clue_policy(train_rows, alpha=alpha)
    packets = build_latent_clue_packets(bundle, split_name=split_name, top_k=card_top_k)
    rows = score_policy_packets(packets, policy)
    if blend_with_stats_weight is None:
        return [dict(row, score_source="policy") for row in rows]
    policy_values = _normalise_scores([_safe_float(row, "score") for row in rows])
    stats_values = _normalise_scores([_sigmoid(6.0 * _safe_float(packet, "corr_margin")) for packet in packets])
    weight = min(max(float(blend_with_stats_weight), 0.0), 1.0)
    mode = blend_mode.strip().lower()
    blended_rows: list[dict[str, str]] = []
    for row, policy_value, stats_value in zip(rows, policy_values, stats_values, strict=True):
        if mode in {"linear", "minmax", "fused"}:
            score = (1.0 - weight) * stats_value + weight * policy_value
            score_source = "policy_fused"
        elif mode in {"safe", "safe_residual", "residual"}:
            score = stats_value + weight * policy_value
            score_source = "policy_safe"
        else:
            raise ValueError("blend_mode must be 'minmax' or 'safe_residual'.")
        blended = dict(row)
        blended["support_score"] = f"{score:.6f}"
        blended["rank_score"] = f"{score:.6f}"
        blended["score"] = f"{score:.6f}"
        blended["score_source"] = score_source
        blended_rows.append(blended)
    return blended_rows


def resolve_sources(values: list[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_SOURCE_LABELS)
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
    prune_soft_scores: bool = False,
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
        if prune_soft_scores:
            dataset["discovery_score_soft_selection"] = "selected"
        else:
            dataset.pop("discovery_score_soft_selection", None)
    elif label == "heuristic":
        dataset["causal_mask_strategy"] = str(dataset.get("causal_mask_strategy", "label_minus_env_correlation"))
        dataset["causal_mask_top_k"] = int(top_k)
        dataset.pop("discovery_scores_path", None)
        dataset.pop("discovery_score_threshold", None)
        dataset.pop("discovery_score_top_k", None)
        dataset.pop("discovery_score_soft_selection", None)
    elif label == "random":
        dataset["causal_mask_strategy"] = "random_top_k"
        dataset["causal_mask_top_k"] = int(top_k)
        dataset["causal_mask_random_seed"] = int(config.get("seed", 0))
        dataset.pop("discovery_scores_path", None)
        dataset.pop("discovery_score_threshold", None)
        dataset.pop("discovery_score_top_k", None)
        dataset.pop("discovery_score_soft_selection", None)
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
    prune_soft_scores: bool = False,
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
                candidate = build_downstream_candidate(
                    base_config,
                    label=label,
                    top_k=top_k,
                    score_path=score_paths.get(label),
                    prune_soft_scores=prune_soft_scores,
                )
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
    parser.add_argument("--sources", action="append", default=[], help="Source label(s) to emit/run: stats, language, image, fused, bridge, bridge_fused, bridge_gated, pairwise_bridge, pairwise_bridge_fused, policy, policy_fused, policy_safe. Can be comma-separated or repeated.")
    parser.add_argument("--out-dir", default="outputs/dfr_sweeps/clue_fusion", help="Directory for cards, clues, scores, ablations, and optional downstream rows.")
    parser.add_argument("--card-top-k", type=int, default=16, help="Top and bottom activation count per feature card.")
    parser.add_argument("--bridge-input-dir", default="outputs/dfr_sweeps/llm_clue_fixture_experiments", help="Fixture trace directory used when source=bridge.")
    parser.add_argument("--bridge-alpha", type=float, default=10.0)
    parser.add_argument("--bridge-fused-weight", type=float, default=0.2, help="Weight on normalized bridge score for source=bridge_fused; the remainder is stats score.")
    parser.add_argument("--bridge-exclude-dataset", action="append", default=["waterbirds"], help="Dataset-name substring to exclude from bridge training. Can be repeated.")
    parser.add_argument("--policy-input-dir", default="outputs/dfr_sweeps/llm_clue_fixture_experiments_20260502_refreshed", help="Fixture trace directory used when source=policy/policy_fused/policy_safe.")
    parser.add_argument("--policy-alpha", type=float, default=10.0)
    parser.add_argument("--policy-fused-weight", type=float, default=0.5, help="Weight on normalized policy score for source=policy_fused/policy_safe.")
    parser.add_argument("--policy-exclude-dataset", action="append", default=["waterbirds"], help="Dataset-name substring to exclude from policy training. Can be repeated.")
    parser.add_argument("--run-downstream", action="store_true", help="Run downstream top-k DFR candidates after writing source score files.")
    parser.add_argument("--prune-soft-scores", action="store_true", help="For discovery-score candidates, zero soft scores outside the selected top-k support.")
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
    image_clues = build_image_prototype_clue_rows(cards)
    clue_rows = merge_external_clue_rows(
        build_feature_clue_rows(bundle, split_name=args.split),
        [*language_clues, *image_clues],
    )

    cards_path = out_dir / "feature_cards.csv"
    language_path = out_dir / "language_clues.csv"
    image_path = out_dir / "image_prototype_clues.csv"
    clue_path = out_dir / "merged_clues.csv"
    write_csv_rows(cards_path, cards)
    write_csv_rows(language_path, language_clues)
    write_csv_rows(image_path, image_clues)
    write_csv_rows(clue_path, clue_rows)

    score_paths: dict[str, Path] = {}
    for source in sources:
        score_path = out_dir / f"scores_{source}.csv"
        if source in {"bridge", "bridge_fused", "bridge_gated"}:
            rows = build_bridge_score_rows(
                bundle,
                bridge_input_dir=Path(args.bridge_input_dir),
                alpha=float(args.bridge_alpha),
                exclude_datasets=_unique_strings(list(args.bridge_exclude_dataset)),
                split_name=args.split,
                card_top_k=args.card_top_k,
                blend_with_stats_weight=float(args.bridge_fused_weight) if source in {"bridge_fused", "bridge_gated"} else None,
                blend_mode="gated" if source == "bridge_gated" else "linear",
            )
        elif source in {"pairwise_bridge", "pairwise_bridge_fused"}:
            rows = build_pairwise_bridge_score_rows(
                bundle,
                bridge_input_dir=Path(args.bridge_input_dir),
                alpha=float(args.bridge_alpha),
                exclude_datasets=_unique_strings(list(args.bridge_exclude_dataset)),
                split_name=args.split,
                card_top_k=args.card_top_k,
                blend_with_stats_weight=float(args.bridge_fused_weight) if source == "pairwise_bridge_fused" else None,
            )
        elif source in {"policy", "policy_fused", "policy_safe"}:
            rows = build_policy_score_rows(
                bundle,
                policy_input_dir=Path(args.policy_input_dir),
                alpha=float(args.policy_alpha),
                exclude_datasets=_unique_strings(list(args.policy_exclude_dataset)),
                split_name=args.split,
                card_top_k=args.card_top_k,
                blend_with_stats_weight=float(args.policy_fused_weight) if source in {"policy_fused", "policy_safe"} else None,
                blend_mode="safe_residual" if source == "policy_safe" else "minmax",
            )
        else:
            rows = build_source_score_rows(clue_rows, source)
        write_csv_rows(score_path, rows)
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
        "bridge_input_dir": str(args.bridge_input_dir),
        "bridge_alpha": float(args.bridge_alpha),
        "bridge_fused_weight": float(args.bridge_fused_weight),
        "bridge_exclude_dataset": _unique_strings(list(args.bridge_exclude_dataset)),
        "policy_input_dir": str(args.policy_input_dir),
        "policy_alpha": float(args.policy_alpha),
        "policy_fused_weight": float(args.policy_fused_weight),
        "policy_exclude_dataset": _unique_strings(list(args.policy_exclude_dataset)),
        "official_dfr_num_retrains": args.official_dfr_num_retrains,
        "training_device": args.training_device,
        "cards": str(cards_path),
        "language_clues": str(language_path),
        "image_prototype_clues": str(image_path),
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
            prune_soft_scores=args.prune_soft_scores,
        )
        downstream_path = out_dir / "downstream_results.csv"
        write_csv_rows(downstream_path, downstream_rows)
        manifest["downstream_results"] = str(downstream_path)
        manifest["prune_soft_scores"] = bool(args.prune_soft_scores)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()

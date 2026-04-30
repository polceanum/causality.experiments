from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import statistics
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
from causality_experiments.run import run_experiment
from scripts.report_clue_source_ablation import summarize_source_ablation
from scripts.run_waterbirds_clue_fusion_sweep import (
    build_downstream_candidate,
    build_source_score_rows,
    resolve_sources,
    with_dataset_path,
    with_runtime_overrides,
)


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _compact_metrics(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    config = payload["config"]
    return {
        "config": config.get("name", ""),
        "method": config.get("method", {}).get("kind", ""),
        "run": run_dir.name,
        "val_wga": float(metrics.get("val/worst_group_accuracy", 0.0)),
        "test_wga": float(metrics.get("test/worst_group_accuracy", 0.0)),
        "val_acc": float(metrics.get("val/accuracy", 0.0)),
        "test_acc": float(metrics.get("test/accuracy", 0.0)),
    }


def _seed_config(base: dict[str, Any], *, label: str, seed: int, output_root: Path) -> dict[str, Any]:
    config = deepcopy(base)
    config["seed"] = int(seed)
    config["name"] = f"{label}_seed{seed}"
    config["output_dir"] = str(output_root)
    return config


def _run_config(config: dict[str, Any], tmp_root: Path, output_root: Path) -> Path:
    config_path = tmp_root / f"{config['name']}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return run_experiment(config_path, output_root)


def _summary(rows: list[dict[str, Any]], *, baseline_label: str, min_mean_delta: float, min_mean_wga: float) -> dict[str, Any]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row["label"]), []).append(row)
    baseline_by_seed = {int(row["seed"]): float(row["test_wga"]) for row in by_label.get(baseline_label, [])}
    candidates: list[dict[str, Any]] = []
    for label, items in sorted(by_label.items()):
        wgas = [float(item["test_wga"]) for item in items]
        val_wgas = [float(item["val_wga"]) for item in items]
        deltas = [
            float(item["test_wga"]) - baseline_by_seed[int(item["seed"])]
            for item in items
            if int(item["seed"]) in baseline_by_seed and label != baseline_label
        ]
        mean_delta = statistics.mean(deltas) if deltas else 0.0
        candidates.append(
            {
                "label": label,
                "count": len(items),
                "mean_test_wga": statistics.mean(wgas),
                "min_test_wga": min(wgas),
                "max_test_wga": max(wgas),
                "std_test_wga": statistics.pstdev(wgas) if len(wgas) > 1 else 0.0,
                "mean_val_wga": statistics.mean(val_wgas),
                "paired_deltas": deltas,
                "mean_delta_to_baseline": mean_delta if label != baseline_label else None,
                "min_delta_to_baseline": min(deltas) if deltas else None,
                "non_negative_seed_count": sum(delta >= 0.0 for delta in deltas),
                "passes_promotion_gate": (
                    label != baseline_label
                    and mean_delta >= min_mean_delta
                    and statistics.mean(wgas) >= min_mean_wga
                    and len(deltas) == len(baseline_by_seed)
                ),
            }
        )
    candidates.sort(
        key=lambda item: (
            bool(item["passes_promotion_gate"]),
            float(item["mean_delta_to_baseline"] or 0.0),
            float(item["mean_test_wga"]),
        ),
        reverse=True,
    )
    return {
        "baseline_label": baseline_label,
        "promotion_gate": {"min_mean_delta": min_mean_delta, "min_mean_wga": min_mean_wga},
        "candidates": candidates,
    }


def write_clue_artifacts(config: dict[str, Any], *, split: str, card_top_k: int, sources: list[str], top_k_values: list[int], out_dir: Path) -> dict[str, Path]:
    bundle = load_dataset(config)
    cards = build_feature_cards(bundle, split_name=split, top_k=card_top_k)
    language_clues = build_language_clue_rows(cards, domain="waterbirds")
    image_clues = build_image_prototype_clue_rows(cards)
    clue_rows = merge_external_clue_rows(build_feature_clue_rows(bundle, split_name=split), [*language_clues, *image_clues])

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
        write_csv_rows(score_path, build_source_score_rows(clue_rows, source))
        score_paths[source] = score_path

    ablation_path = out_dir / "source_ablation.csv"
    write_csv_rows(
        ablation_path,
        summarize_source_ablation(clue_path, [(label, path) for label, path in score_paths.items()], top_k_values=top_k_values, reference_label="stats"),
    )
    return {"cards": cards_path, "language_clues": language_path, "image_prototype_clues": image_path, "merged_clues": clue_path, "source_ablation": ablation_path, **{f"scores_{label}": path for label, path in score_paths.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", default="configs/benchmarks/waterbirds_features_official_dfr_val_tr.yaml")
    parser.add_argument("--candidate-config", default="configs/benchmarks/waterbirds_features_official_causal_dfr_soft.yaml")
    parser.add_argument("--dataset-path", default="data/waterbirds/features_official_erm_official_repro.csv")
    parser.add_argument("--seeds", default="101,102,103")
    parser.add_argument("--top-k", type=int, action="append", default=[])
    parser.add_argument("--sources", action="append", default=[])
    parser.add_argument("--split", default="train")
    parser.add_argument("--card-top-k", type=int, default=16)
    parser.add_argument("--training-device", default="cpu")
    parser.add_argument("--output-root", default="outputs/runs")
    parser.add_argument("--out-dir", default="outputs/dfr_sweeps/clue_seed_stability")
    parser.add_argument("--include-heuristic", action="store_true")
    parser.add_argument("--include-random", action="store_true")
    parser.add_argument("--prune-soft-scores", action="store_true")
    parser.add_argument("--min-mean-delta", type=float, default=0.003)
    parser.add_argument("--min-mean-wga", type=float, default=0.935)
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    top_k_values = args.top_k or [64, 128]
    sources = resolve_sources(args.sources)
    output_root = Path(args.output_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_base = with_runtime_overrides(
        with_dataset_path(load_config(Path(args.baseline_config)), args.dataset_path),
        training_device=args.training_device,
    )
    candidate_base = with_runtime_overrides(
        with_dataset_path(load_config(Path(args.candidate_config)), args.dataset_path),
        training_device=args.training_device,
    )
    candidate_base["output_dir"] = str(output_root)
    baseline_base["output_dir"] = str(output_root)

    artifact_paths = write_clue_artifacts(
        candidate_base,
        split=args.split,
        card_top_k=args.card_top_k,
        sources=sources,
        top_k_values=top_k_values,
        out_dir=out_dir,
    )
    score_paths = {source: artifact_paths[f"scores_{source}"] for source in sources}

    rows: list[dict[str, Any]] = []
    baseline_label = Path(args.baseline_config).stem
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        for seed in seeds:
            baseline = _seed_config(baseline_base, label=baseline_label, seed=seed, output_root=output_root)
            baseline_metrics = _compact_metrics(_run_config(baseline, tmp_root, output_root))
            rows.append({"label": baseline_label, "seed": seed, "top_k": "", "source": "baseline", **baseline_metrics})
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
            labels = list(score_paths)
            if args.include_heuristic:
                labels.append("heuristic")
            if args.include_random:
                labels.append("random")
            for label in labels:
                for top_k in top_k_values:
                    candidate = build_downstream_candidate(
                        candidate_base,
                        label=label,
                        top_k=top_k,
                        score_path=score_paths.get(label),
                        prune_soft_scores=args.prune_soft_scores,
                    )
                    candidate_label = f"{Path(args.candidate_config).stem}_{label}_top{top_k}"
                    candidate = _seed_config(candidate, label=candidate_label, seed=seed, output_root=output_root)
                    metrics = _compact_metrics(_run_config(candidate, tmp_root, output_root))
                    rows.append({"label": candidate_label, "seed": seed, "top_k": top_k, "source": label, **metrics})
                    print(json.dumps(rows[-1], sort_keys=True), flush=True)

    rows_path = out_dir / "seed_stability_rows.csv"
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(rows, baseline_label=baseline_label, min_mean_delta=args.min_mean_delta, min_mean_wga=args.min_mean_wga)
    summary_path = out_dir / "seed_stability_summary.json"
    manifest = {
        "baseline_config": args.baseline_config,
        "candidate_config": args.candidate_config,
        "dataset_path": args.dataset_path,
        "seeds": seeds,
        "top_k": top_k_values,
        "sources": sources,
        "include_heuristic": bool(args.include_heuristic),
        "include_random": bool(args.include_random),
        "prune_soft_scores": bool(args.prune_soft_scores),
        "rows": str(rows_path),
        "summary": str(summary_path),
        "artifacts": {key: str(path) for key, path in artifact_paths.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "summary": str(summary_path), "rows": str(rows_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

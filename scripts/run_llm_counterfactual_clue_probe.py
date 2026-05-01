from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.clues import build_feature_cards, write_csv_rows
from causality_experiments.config import load_config
from causality_experiments.counterfactual_clue_tests import clue_rows_from_test_results, execute_clue_tests
from causality_experiments.data import load_dataset
from causality_experiments.discovery import build_feature_clue_rows
from causality_experiments.latent_clue_packets import build_latent_clue_packets, packets_to_jsonl
from causality_experiments.llm_clue_bridge import build_bridge_training_rows
from causality_experiments.llm_clue_planner import MockCluePlannerBackend, plan_from_backend
from causality_experiments.methods import fit_method


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")


def _hypothesis_clue_rows(hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in hypotheses:
        confidence = float(row.get("confidence", 0.0) or 0.0)
        hypothesis_type = str(row.get("hypothesis_type", "uncertain"))
        rows.append(
            {
                "dataset": "",
                "feature_name": row.get("feature_name", ""),
                "llm_hypothesis_type": hypothesis_type,
                "llm_confidence": f"{confidence:.6f}",
                "llm_reason_code": row.get("reason_code", ""),
                "llm_untested": "1",
                "test_passed_control": "0",
            }
        )
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_rows_from_clues(rows: list[dict[str, Any]], *, source: str) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        if source == "stats":
            score = _safe_float(row.get("soft_causal_target"), _safe_float(row.get("corr_margin")))
        elif source == "random":
            feature_index = int(_safe_float(row.get("feature_index")))
            score = ((feature_index * 1103515245 + 12345) % 1000000) / 1000000.0
        elif source == "llm_tested":
            passed = str(row.get("test_passed_control", "0")).strip().lower() in {"1", "true", "yes"}
            score = _safe_float(row.get("llm_confidence")) if passed else 0.0
        else:
            raise ValueError(f"Unknown score source {source!r}.")
        output.append(
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
    return output


def _compare_score_rows(feature_clues: list[dict[str, Any]], score_sets: dict[str, list[dict[str, str]]], top_k_values: list[int]) -> list[dict[str, str]]:
    clues_by_name = {str(row.get("feature_name", "")): row for row in feature_clues}
    rows: list[dict[str, str]] = []
    for label, scores in score_sets.items():
        ranked = sorted(scores, key=lambda row: _safe_float(row.get("score")), reverse=True)
        for top_k in top_k_values:
            selected = [row for row in ranked if str(row.get("feature_name", "")) in clues_by_name][:top_k]
            if not selected:
                continue
            joined = [clues_by_name[str(row["feature_name"])] for row in selected]
            causal_values = [_safe_float(row.get("causal_target"), 0.0) for row in joined]
            corr_margins = [_safe_float(row.get("corr_margin"), 0.0) for row in joined]
            rows.append(
                {
                    "label": label,
                    "top_k": str(len(selected)),
                    "mean_causal_target": f"{sum(causal_values) / max(len(causal_values), 1):.6f}",
                    "mean_corr_margin": f"{sum(corr_margins) / max(len(corr_margins), 1):.6f}",
                    "selected_features": ";".join(str(row["feature_name"]) for row in selected),
                }
            )
    return rows


def run_llm_counterfactual_clue_probe(
    *,
    config_path: Path,
    out_dir: Path,
    split: str = "train",
    card_top_k: int = 8,
    max_packets: int = 16,
    llm_backend: str = "mock",
    execute_tests: bool = True,
    test_split: str = "test",
) -> dict[str, str]:
    if llm_backend != "mock":
        raise ValueError("Only --llm-backend mock is implemented in the first offline slice.")
    config = load_config(config_path)
    bundle = load_dataset(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    cards = build_feature_cards(bundle, split_name=split, top_k=card_top_k)
    feature_clues = build_feature_clue_rows(bundle, split_name=split)
    packets = build_latent_clue_packets(bundle, split_name=split, top_k=card_top_k, max_packets=max_packets)
    plan = plan_from_backend(packets, MockCluePlannerBackend(), max_packets=max_packets)
    hypotheses = [asdict(hypothesis) for hypothesis in plan.hypotheses]
    tests = [asdict(test) for test in plan.tests]
    test_results: list[dict[str, Any]] = []
    if execute_tests:
        model = fit_method(bundle, config)
        test_results = execute_clue_tests(bundle, plan.tests, packets=packets, model=model, split_name=test_split)
    trace_rows = build_bridge_training_rows(packets, plan.hypotheses, plan.tests, test_results)
    clue_rows = clue_rows_from_test_results(test_results) if test_results else _hypothesis_clue_rows(hypotheses)
    llm_score_rows = _score_rows_from_clues(clue_rows, source="llm_tested")
    stats_score_rows = _score_rows_from_clues(feature_clues, source="stats")
    random_score_rows = _score_rows_from_clues(feature_clues, source="random")
    top_k_values = sorted({1, min(2, max(1, bundle.input_dim)), min(4, max(1, bundle.input_dim))})
    comparison_rows = _compare_score_rows(
        feature_clues,
        {"llm_tested": llm_score_rows, "stats": stats_score_rows, "random": random_score_rows},
        top_k_values=top_k_values,
    )

    cards_path = out_dir / "feature_cards.csv"
    feature_clues_path = out_dir / "feature_clues.csv"
    packets_path = out_dir / "latent_clue_packets.jsonl"
    hypotheses_path = out_dir / "hypotheses.jsonl"
    tests_path = out_dir / "test_specs.jsonl"
    results_path = out_dir / "test_results.csv"
    traces_path = out_dir / "training_traces.jsonl"
    clues_path = out_dir / "llm_clues.csv"
    llm_scores_path = out_dir / "scores_llm_tested.csv"
    stats_scores_path = out_dir / "scores_stats.csv"
    random_scores_path = out_dir / "scores_random.csv"
    comparison_path = out_dir / "baseline_comparison.csv"
    manifest_path = out_dir / "manifest.json"

    write_csv_rows(cards_path, cards)
    write_csv_rows(feature_clues_path, feature_clues)
    packets_path.write_text(packets_to_jsonl(packets), encoding="utf-8")
    _write_jsonl(hypotheses_path, hypotheses)
    _write_jsonl(tests_path, tests)
    if test_results:
        write_csv_rows(results_path, test_results)
    _write_jsonl(traces_path, trace_rows)
    write_csv_rows(clues_path, clue_rows)
    write_csv_rows(llm_scores_path, llm_score_rows)
    write_csv_rows(stats_scores_path, stats_score_rows)
    write_csv_rows(random_scores_path, random_score_rows)
    write_csv_rows(comparison_path, comparison_rows)
    manifest = {
        "config": str(config_path),
        "dataset": bundle.name,
        "split": split,
        "card_top_k": int(card_top_k),
        "max_packets": int(max_packets),
        "llm_backend": llm_backend,
        "planner_backend": plan.backend,
        "planner_fallback": bool(plan.fallback),
        "execute_tests": bool(execute_tests),
        "test_split": test_split,
        "cards": str(cards_path),
        "feature_clues": str(feature_clues_path),
        "latent_clue_packets": str(packets_path),
        "hypotheses": str(hypotheses_path),
        "test_specs": str(tests_path),
        "test_results": str(results_path) if test_results else "",
        "training_traces": str(traces_path),
        "llm_clues": str(clues_path),
        "scores_llm_tested": str(llm_scores_path),
        "scores_stats": str(stats_scores_path),
        "scores_random": str(random_scores_path),
        "baseline_comparison": str(comparison_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return {key: str(value) for key, value in manifest.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="outputs/dfr_sweeps/llm_counterfactual_clue_probe")
    parser.add_argument("--split", default="train")
    parser.add_argument("--card-top-k", type=int, default=8)
    parser.add_argument("--max-packets", type=int, default=16)
    parser.add_argument("--llm-backend", default="mock", choices=("mock",))
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--dry-run", action="store_true", help="Only write planner proposals; skip deterministic test execution.")
    args = parser.parse_args()
    manifest = run_llm_counterfactual_clue_probe(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        split=args.split,
        card_top_k=args.card_top_k,
        max_packets=args.max_packets,
        llm_backend=args.llm_backend,
        execute_tests=not bool(args.dry_run),
        test_split=args.test_split,
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

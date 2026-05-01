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

    cards_path = out_dir / "feature_cards.csv"
    packets_path = out_dir / "latent_clue_packets.jsonl"
    hypotheses_path = out_dir / "hypotheses.jsonl"
    tests_path = out_dir / "test_specs.jsonl"
    results_path = out_dir / "test_results.csv"
    traces_path = out_dir / "training_traces.jsonl"
    clues_path = out_dir / "llm_clues.csv"
    manifest_path = out_dir / "manifest.json"

    write_csv_rows(cards_path, cards)
    packets_path.write_text(packets_to_jsonl(packets), encoding="utf-8")
    _write_jsonl(hypotheses_path, hypotheses)
    _write_jsonl(tests_path, tests)
    if test_results:
        write_csv_rows(results_path, test_results)
    _write_jsonl(traces_path, trace_rows)
    write_csv_rows(clues_path, clue_rows)
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
        "latent_clue_packets": str(packets_path),
        "hypotheses": str(hypotheses_path),
        "test_specs": str(tests_path),
        "test_results": str(results_path) if test_results else "",
        "training_traces": str(traces_path),
        "llm_clues": str(clues_path),
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

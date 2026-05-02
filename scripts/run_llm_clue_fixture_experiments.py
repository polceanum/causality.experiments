from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_llm_counterfactual_clue_probe import run_llm_counterfactual_clue_probe


def run_fixture_experiments(
    *,
    config_dir: Path,
    out_dir: Path,
    split: str,
    card_top_k: int,
    max_packets: int,
    llm_backend: str,
    execute_tests: bool,
    test_split: str,
) -> dict[str, Any]:
    configs = sorted(config_dir.glob("*.yaml"))
    if not configs:
        raise ValueError(f"No fixture configs found under {config_dir}.")
    runs: list[dict[str, str]] = []
    for config_path in configs:
        fixture_out_dir = out_dir / config_path.stem
        manifest = run_llm_counterfactual_clue_probe(
            config_path=config_path,
            out_dir=fixture_out_dir,
            split=split,
            card_top_k=card_top_k,
            max_packets=max_packets,
            llm_backend=llm_backend,
            execute_tests=execute_tests,
            test_split=test_split,
        )
        runs.append({"config": str(config_path), "out_dir": str(fixture_out_dir), **manifest})
        print(json.dumps(runs[-1], sort_keys=True), flush=True)
    summary = {
        "config_dir": str(config_dir),
        "out_dir": str(out_dir),
        "split": split,
        "card_top_k": int(card_top_k),
        "max_packets": int(max_packets),
        "llm_backend": llm_backend,
        "execute_tests": bool(execute_tests),
        "test_split": test_split,
        "runs": runs,
    }
    summary_path = out_dir / "manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["manifest"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="configs/experiments")
    parser.add_argument("--out-dir", default="outputs/dfr_sweeps/llm_clue_fixture_experiments")
    parser.add_argument("--split", default="train")
    parser.add_argument("--card-top-k", type=int, default=8)
    parser.add_argument("--max-packets", type=int, default=16)
    parser.add_argument("--llm-backend", default="mock", choices=("mock",))
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--dry-run", action="store_true", help="Only write planner proposals; skip deterministic test execution.")
    args = parser.parse_args()
    summary = run_fixture_experiments(
        config_dir=Path(args.config_dir),
        out_dir=Path(args.out_dir),
        split=str(args.split),
        card_top_k=int(args.card_top_k),
        max_packets=int(args.max_packets),
        llm_backend=str(args.llm_backend),
        execute_tests=not bool(args.dry_run),
        test_split=str(args.test_split),
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
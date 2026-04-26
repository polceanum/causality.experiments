from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import tempfile
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.config import load_config
from causality_experiments.run import run_experiment
from scripts.tune_discovery_mask import _build_candidate


def _specs(
    top_k: int,
    utility_scores: str,
    *,
    include_scored: bool,
    score_weight: float,
    include_utility: bool,
) -> list[dict[str, str | int | float | None]]:
    specs: list[dict[str, str | int | float | None]] = [
        {
            "label": f"grouped_discovery_top{top_k}",
            "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_gate_nuisance0p9.yaml",
            "variant": "discovery_full",
            "score_path": "outputs/runs/waterbirds-feature-discovery-scores.csv",
            "top_k": top_k,
        },
    ]
    if include_scored:
        specs.append(
            {
                "label": f"grouped_scored_top{top_k}_w{str(score_weight).replace('.', 'p')}",
                "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_scored_gate_nuisance0p9.yaml",
                "variant": "discovery_full",
                "score_path": "outputs/runs/waterbirds-feature-discovery-scores.csv",
                "top_k": top_k,
                "score_weight": score_weight,
            }
        )
    if include_utility:
        specs.append(
            {
                "label": f"grouped_utility_top{top_k}",
                "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_gate_nuisance0p9.yaml",
                "variant": "discovery_full",
                "score_path": utility_scores,
                "top_k": top_k,
            }
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-scored", action="store_true")
    parser.add_argument("--include-utility", action="store_true")
    parser.add_argument("--score-weight", type=float, default=0.5)
    parser.add_argument(
        "--utility-scores",
        default="outputs/runs/waterbirds-feature-discovery-scores-utility.csv",
    )
    args = parser.parse_args()

    for spec in _specs(
        args.top_k,
        args.utility_scores,
        include_scored=args.include_scored,
        score_weight=args.score_weight,
        include_utility=args.include_utility,
    ):
        base = load_config(Path(str(spec["base"])))
        config = deepcopy(base)
        config.setdefault("training", {})["device"] = args.device
        if "score_weight" in spec:
            config.setdefault("method", {})["input_gate_score_weight"] = float(spec["score_weight"])
        config = _build_candidate(
            config,
            top_k=int(spec["top_k"]),
            variant=str(spec["variant"]),
            score_path=str(spec["score_path"]),
        )
        config.setdefault("training", {})["device"] = args.device
        if "score_weight" in spec:
            config.setdefault("method", {})["input_gate_score_weight"] = float(spec["score_weight"])
        tmp_config = Path(tempfile.mkdtemp(prefix="waterbirds-grouped-gate-")) / "config.yaml"
        tmp_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run_dir = run_experiment(tmp_config)
        print(run_dir)


if __name__ == "__main__":
    main()
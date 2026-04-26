from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.config import load_config
from causality_experiments.run import run_experiment
from scripts.run_instability_jtt_sweep import _promotion_decision
from scripts.tune_discovery_mask import _build_candidate


def _specs(
    *,
    top_k: int,
    full_scores: str,
    utility_scores: str,
    include_controls: bool,
) -> list[dict[str, str | int | None]]:
    specs: list[dict[str, str | int | None]] = [
        {
            "label": "fixed_base",
            "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_gated_nuisance0p9.yaml",
            "variant": None,
            "score_path": None,
            "top_k": None,
        },
        {
            "label": f"fixed_discovery_top{top_k}",
            "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_gated_nuisance0p9.yaml",
            "variant": "discovery_full",
            "score_path": full_scores,
            "top_k": top_k,
        },
        {
            "label": f"fixed_instability_jtt_discovery_top{top_k}",
            "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_fixed_instability_jtt_gated_nuisance0p9.yaml",
            "variant": "discovery_full",
            "score_path": full_scores,
            "top_k": top_k,
        },
        {
            "label": f"learned_discovery_top{top_k}",
            "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_gate_nuisance0p9.yaml",
            "variant": "discovery_full",
            "score_path": full_scores,
            "top_k": top_k,
        },
        {
            "label": f"grouped_discovery_top{top_k}",
            "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_gate_nuisance0p9.yaml",
            "variant": "discovery_full",
            "score_path": full_scores,
            "top_k": top_k,
        },
        {
            "label": f"grouped_scored_discovery_top{top_k}",
            "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_scored_gate_nuisance0p9.yaml",
            "variant": "discovery_full",
            "score_path": full_scores,
            "top_k": top_k,
        },
        {
            "label": f"grouped_conditioned_discovery_top{top_k}",
            "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_conditioned_gate_nuisance0p9.yaml",
            "variant": "discovery_full",
            "score_path": full_scores,
            "top_k": top_k,
        },
        {
            "label": f"grouped_contextual_discovery_top{top_k}",
            "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_contextual_gate_nuisance0p9.yaml",
            "variant": "discovery_full",
            "score_path": full_scores,
            "top_k": top_k,
        },
        {
            "label": f"grouped_representation_discovery_top{top_k}",
            "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_representation_gate_nuisance0p9.yaml",
            "variant": "discovery_full",
            "score_path": full_scores,
            "top_k": top_k,
        },
        {
            "label": f"grouped_disagreement_discovery_top{top_k}",
            "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_disagreement_gate_nuisance0p9.yaml",
            "variant": "discovery_full",
            "score_path": full_scores,
            "top_k": top_k,
        },
        {
            "label": f"grouped_replay_discovery_top{top_k}",
            "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_replay_gate_nuisance0p9.yaml",
            "variant": "discovery_full",
            "score_path": full_scores,
            "top_k": top_k,
        },
        {
            "label": f"grouped_instability_jtt_discovery_top{top_k}",
            "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_instability_jtt_gate_nuisance0p9.yaml",
            "variant": "discovery_full",
            "score_path": full_scores,
            "top_k": top_k,
        },
        {
            "label": f"grouped_stable_instability_jtt_discovery_top{top_k}",
            "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_stable_instability_jtt_gate_nuisance0p9.yaml",
            "variant": "discovery_full",
            "score_path": full_scores,
            "top_k": top_k,
        },
        {
            "label": f"grouped_loss_weighted_instability_jtt_discovery_top{top_k}",
            "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_loss_weighted_instability_jtt_gate_nuisance0p9.yaml",
            "variant": "discovery_full",
            "score_path": full_scores,
            "top_k": top_k,
        },
        {
            "label": f"grouped_loss_delta_instability_jtt_discovery_top{top_k}",
            "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_loss_delta_instability_jtt_gate_nuisance0p9.yaml",
            "variant": "discovery_full",
            "score_path": full_scores,
            "top_k": top_k,
        },
        {
            "label": f"grouped_group_loss_delta_instability_jtt_discovery_top{top_k}",
            "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_group_loss_delta_instability_jtt_gate_nuisance0p9.yaml",
            "variant": "discovery_full",
            "score_path": full_scores,
            "top_k": top_k,
        },
        {
            "label": f"grouped_group_failure_instability_jtt_discovery_top{top_k}",
            "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_group_failure_instability_jtt_gate_nuisance0p9.yaml",
            "variant": "discovery_full",
            "score_path": full_scores,
            "top_k": top_k,
        },
        {
            "label": f"grouped_utility_top{top_k}",
            "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_gate_nuisance0p9.yaml",
            "variant": "discovery_full",
            "score_path": utility_scores,
            "top_k": top_k,
        },
    ]
    if include_controls:
        specs.extend(
            [
                {
                    "label": f"fixed_heuristic_top{top_k}",
                    "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_gated_nuisance0p9.yaml",
                    "variant": "heuristic",
                    "score_path": None,
                    "top_k": top_k,
                },
                {
                    "label": f"fixed_random_top{top_k}",
                    "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_gated_nuisance0p9.yaml",
                    "variant": "random",
                    "score_path": None,
                    "top_k": top_k,
                },
                {
                    "label": f"grouped_heuristic_top{top_k}",
                    "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_gate_nuisance0p9.yaml",
                    "variant": "heuristic",
                    "score_path": None,
                    "top_k": top_k,
                },
                {
                    "label": f"grouped_random_top{top_k}",
                    "base": "configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_gate_nuisance0p9.yaml",
                    "variant": "random",
                    "score_path": None,
                    "top_k": top_k,
                },
            ]
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Epochs for each compact check run.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device override for compact checks. Defaults to cpu for repeatable short runs.",
    )
    parser.add_argument("--top-k", type=int, default=128, help="Top-k support size for discovery/control variants.")
    parser.add_argument(
        "--full-scores",
        default="outputs/runs/waterbirds-feature-discovery-scores.csv",
        help="Discovery score CSV used for the standard learned mask variants.",
    )
    parser.add_argument(
        "--utility-scores",
        default="outputs/runs/waterbirds-feature-discovery-scores-utility.csv",
        help="Utility-aware discovery score CSV used for the grouped utility variant.",
    )
    parser.add_argument(
        "--include-controls",
        action="store_true",
        help="Include matched heuristic and random controls for fixed and grouped consumers.",
    )
    parser.add_argument(
        "--out",
        default="outputs/runs/waterbirds-learned-gate-compact-check.csv",
        help="CSV path for compact comparison results.",
    )
    parser.add_argument("--promotion-min-test-wga", type=float, default=0.68)
    parser.add_argument("--promotion-min-val-wga", type=float, default=0.68)
    parser.add_argument("--promotion-max-test-val-gap", type=float, default=0.03)
    args = parser.parse_args()

    rows: list[dict[str, str | float]] = []
    out_root = Path(args.out).resolve().parent
    out_root.mkdir(parents=True, exist_ok=True)
    for spec in _specs(
        top_k=args.top_k,
        full_scores=args.full_scores,
        utility_scores=args.utility_scores,
        include_controls=args.include_controls,
    ):
        base = load_config(Path(str(spec["base"])))
        config = deepcopy(base)
        config["training"]["epochs"] = args.epochs
        config.setdefault("training", {})["device"] = args.device
        config["output_dir"] = str(out_root)
        config["name"] = config["name"] + "_compact_" + str(spec["label"])
        if spec["variant"] is not None:
            config = _build_candidate(
                config,
                top_k=int(spec["top_k"]),
                variant=str(spec["variant"]),
                score_path=str(spec["score_path"]),
            )
            config["training"]["epochs"] = args.epochs
            config.setdefault("training", {})["device"] = args.device
            config["output_dir"] = str(out_root)
            config["name"] = config["name"] + "_compact"
        tmp_config = Path(tempfile.mkdtemp(prefix="waterbirds-compact-check-")) / "config.yaml"
        tmp_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run_dir = run_experiment(tmp_config)
        payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        test_wga = float(metrics["test/worst_group_accuracy"])
        val_wga = float(metrics["val/worst_group_accuracy"])
        eligible, promotion_score = _promotion_decision(
            test_wga=test_wga,
            val_wga=val_wga,
            min_test_wga=args.promotion_min_test_wga,
            min_val_wga=args.promotion_min_val_wga,
            max_test_val_gap=args.promotion_max_test_val_gap,
        )
        rows.append(
            {
                "label": str(spec["label"]),
                "run": run_dir.name,
                "config": config["name"],
                "test_wga": test_wga,
                "val_wga": val_wga,
                "test_acc": float(metrics["test/accuracy"]),
                "val_acc": float(metrics["val/accuracy"]),
                "promotion_score": promotion_score,
                "eligible_for_promotion": int(eligible),
            }
        )

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)
    for row in rows:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
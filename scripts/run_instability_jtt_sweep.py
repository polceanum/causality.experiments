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
from scripts.tune_discovery_mask import _build_candidate


def _promotion_decision(
    *,
    test_wga: float,
    val_wga: float,
    min_test_wga: float,
    min_val_wga: float,
    max_test_val_gap: float,
) -> tuple[bool, float]:
    promotion_score = min(float(test_wga), float(val_wga))
    gap = abs(float(test_wga) - float(val_wga))
    eligible = (
        float(test_wga) >= float(min_test_wga)
        and float(val_wga) >= float(min_val_wga)
        and gap <= float(max_test_val_gap)
    )
    return eligible, promotion_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_instability_jtt_gate_nuisance0p9.yaml",
    )
    parser.add_argument("--stage1-epochs", type=int, action="append", required=True)
    parser.add_argument("--top-fraction", type=float, action="append", required=True)
    parser.add_argument("--upweight", type=float, action="append", required=True)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--score-mode", default=None)
    parser.add_argument("--promotion-min-test-wga", type=float, default=0.68)
    parser.add_argument("--promotion-min-val-wga", type=float, default=0.68)
    parser.add_argument("--promotion-max-test-val-gap", type=float, default=0.03)
    parser.add_argument(
        "--out",
        default="outputs/runs/waterbirds-instability-jtt-sweep.csv",
    )
    args = parser.parse_args()

    base = load_config(Path(args.base_config))
    rows: list[dict[str, str | float | int]] = []
    for stage1_epochs in args.stage1_epochs:
        for top_fraction in args.top_fraction:
            for upweight in args.upweight:
                config = deepcopy(base)
                config.setdefault("training", {})["epochs"] = args.epochs
                config.setdefault("training", {})["device"] = args.device
                method = config.setdefault("method", {})
                method["counterfactual_instability_stage1_epochs"] = int(stage1_epochs)
                method["counterfactual_instability_passes"] = int(args.passes)
                method["counterfactual_instability_top_fraction"] = float(top_fraction)
                method["counterfactual_instability_upweight"] = float(upweight)
                if args.score_mode is not None:
                    method["counterfactual_instability_score_mode"] = str(args.score_mode)
                config = _build_candidate(
                    config,
                    top_k=args.top_k,
                    variant="discovery_full",
                    score_path="outputs/runs/waterbirds-feature-discovery-scores.csv",
                )
                config.setdefault("training", {})["epochs"] = args.epochs
                config.setdefault("training", {})["device"] = args.device
                method = config.setdefault("method", {})
                method["counterfactual_instability_stage1_epochs"] = int(stage1_epochs)
                method["counterfactual_instability_passes"] = int(args.passes)
                method["counterfactual_instability_top_fraction"] = float(top_fraction)
                method["counterfactual_instability_upweight"] = float(upweight)
                if args.score_mode is not None:
                    method["counterfactual_instability_score_mode"] = str(args.score_mode)
                config["name"] = (
                    config["name"]
                    + f"_stage1e{stage1_epochs}_topf{str(top_fraction).replace('.', 'p')}_upw{str(upweight).replace('.', 'p')}"
                    + "_compact"
                )
                tmp_config = Path(tempfile.mkdtemp(prefix="waterbirds-instability-jtt-sweep-")) / "config.yaml"
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
                        "stage1_epochs": int(stage1_epochs),
                        "top_fraction": float(top_fraction),
                        "upweight": float(upweight),
                        "passes": int(args.passes),
                        "score_mode": str(method.get("counterfactual_instability_score_mode", "mean")),
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)
    eligible_rows = [row for row in rows if int(row["eligible_for_promotion"]) == 1]
    ranked_rows = sorted(
        eligible_rows,
        key=lambda row: (float(row["promotion_score"]), float(row["test_wga"]), float(row["val_wga"])),
        reverse=True,
    )
    if ranked_rows:
        print("eligible_promotion_candidates")
        for row in ranked_rows[:5]:
            print(json.dumps(row, sort_keys=True))
    else:
        print("eligible_promotion_candidates")
        print("[]")
    for row in rows:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--variant", default="discovery_full")
    parser.add_argument(
        "--score-path",
        default="outputs/runs/waterbirds-feature-discovery-scores.csv",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage1-epochs", type=int, default=0)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--top-fraction", type=float, default=0.0)
    parser.add_argument("--upweight", type=float, default=1.0)
    parser.add_argument("--score-mode", default=None)
    parser.add_argument("--name-suffix", default="")
    args = parser.parse_args()

    config = load_config(Path(args.base_config))
    config = deepcopy(config)
    config.setdefault("training", {})["device"] = args.device
    method = config.setdefault("method", {})
    method["counterfactual_instability_stage1_epochs"] = int(args.stage1_epochs)
    method["counterfactual_instability_passes"] = int(args.passes)
    method["counterfactual_instability_top_fraction"] = float(args.top_fraction)
    method["counterfactual_instability_upweight"] = float(args.upweight)
    if args.score_mode is not None:
        method["counterfactual_instability_score_mode"] = str(args.score_mode)

    config = _build_candidate(
        config,
        top_k=args.top_k,
        variant=args.variant,
        score_path=args.score_path,
    )

    config.setdefault("training", {})["device"] = args.device
    method = config.setdefault("method", {})
    method["counterfactual_instability_stage1_epochs"] = int(args.stage1_epochs)
    method["counterfactual_instability_passes"] = int(args.passes)
    method["counterfactual_instability_top_fraction"] = float(args.top_fraction)
    method["counterfactual_instability_upweight"] = float(args.upweight)
    if args.score_mode is not None:
        method["counterfactual_instability_score_mode"] = str(args.score_mode)
    if args.name_suffix:
        config["name"] = config["name"] + str(args.name_suffix)

    tmp_config = Path(tempfile.mkdtemp(prefix="promote-instability-candidate-")) / "config.yaml"
    tmp_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    run_dir = run_experiment(tmp_config)
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    print(run_dir)
    print(json.dumps(payload["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
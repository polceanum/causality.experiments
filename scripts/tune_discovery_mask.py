from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.config import load_config
from causality_experiments.run import run_experiment


def _candidate_name(base_name: str, *, variant: str, top_k: int) -> str:
    return f"{base_name}_{variant}_top{top_k}"


def _build_candidate(
    base_config: dict,
    *,
    top_k: int,
    variant: str,
    score_path: str | None = None,
) -> dict:
    config = deepcopy(base_config)
    config["name"] = _candidate_name(config["name"], variant=variant, top_k=top_k)
    dataset = dict(config.get("dataset", {}))
    if variant in {"discovery_full", "discovery_restricted"}:
        if not score_path:
            raise ValueError(f"{variant} candidates require a discovery score path.")
        dataset["causal_mask_strategy"] = "discovery_scores"
        dataset["discovery_scores_path"] = score_path
        dataset["discovery_score_threshold"] = 2.0
        dataset["discovery_score_top_k"] = top_k
    elif variant == "heuristic":
        dataset["causal_mask_strategy"] = str(dataset.get("causal_mask_strategy", "label_minus_env_correlation"))
        dataset["causal_mask_top_k"] = top_k
        dataset.pop("discovery_scores_path", None)
        dataset.pop("discovery_score_threshold", None)
        dataset.pop("discovery_score_top_k", None)
    elif variant == "random":
        dataset["causal_mask_strategy"] = "random_top_k"
        dataset["causal_mask_top_k"] = top_k
        dataset["causal_mask_random_seed"] = int(config.get("seed", 0))
        dataset.pop("discovery_scores_path", None)
        dataset.pop("discovery_score_threshold", None)
        dataset.pop("discovery_score_top_k", None)
    else:
        raise ValueError(f"Unknown candidate variant {variant!r}.")
    config["dataset"] = dataset
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True, help="Benchmark config used as the downstream run template.")
    parser.add_argument("--full-scores", required=True, help="Discovery score CSV without support restriction.")
    parser.add_argument("--restricted-scores", help="Discovery score CSV restricted to the heuristic support.")
    parser.add_argument("--include-heuristic", action="store_true", help="Include a matched-cardinality heuristic-correlation control for each top-k.")
    parser.add_argument("--include-random", action="store_true", help="Include a matched-cardinality random-mask control for each top-k.")
    parser.add_argument("--top-k", type=int, action="append", required=True, help="Candidate top-k value. Can be provided multiple times.")
    parser.add_argument("--out", required=True, help="CSV path for candidate results.")
    args = parser.parse_args()

    base_config = load_config(Path(args.base_config))
    candidates: list[dict[str, str | float | int]] = []
    variant_specs: list[tuple[str, str | None]] = [("discovery_full", args.full_scores)]
    if args.restricted_scores:
        variant_specs.append(("discovery_restricted", args.restricted_scores))
    if args.include_heuristic:
        variant_specs.append(("heuristic", None))
    if args.include_random:
        variant_specs.append(("random", None))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        for variant, score_path in variant_specs:
            for top_k in args.top_k:
                candidate = _build_candidate(base_config, score_path=score_path, top_k=top_k, variant=variant)
                config_path = tmp_root / f"{candidate['name']}.yaml"
                config_path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
                run_dir = run_experiment(config_path)
                payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                metrics = payload["metrics"]
                candidates.append(
                    {
                        "config": candidate["name"],
                        "variant": variant,
                        "top_k": top_k,
                        "run": run_dir.name,
                        "val_wga": metrics.get("val/worst_group_accuracy", float("nan")),
                        "test_wga": metrics.get("test/worst_group_accuracy", float("nan")),
                        "val_acc": metrics.get("val/accuracy", float("nan")),
                        "test_acc": metrics.get("test/accuracy", float("nan")),
                    }
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0].keys()))
        writer.writeheader()
        writer.writerows(candidates)
    print(out_path)


if __name__ == "__main__":
    main()
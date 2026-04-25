from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.config import load_config
from causality_experiments.data import load_dataset
from causality_experiments.discovery import (
    build_discovery_model,
    build_feature_clue_rows,
    combine_discovery_scores,
)


def _allowed_feature_names(config_path: str | None) -> set[str] | None:
    if not config_path:
        return None
    bundle = load_dataset(load_config(Path(config_path)))
    if bundle.causal_mask is None:
        raise ValueError("Restrict-to config did not produce a causal mask.")
    feature_columns = list((bundle.metadata or {}).get("feature_columns", []))
    return {
        feature_name
        for feature_name, keep in zip(feature_columns, bundle.causal_mask.tolist(), strict=True)
        if keep > 0.0
    }


def _apply_support_restriction(
    scored_rows: list[dict[str, str]],
    allowed_features: set[str] | None,
    outside_score: float,
) -> list[dict[str, str]]:
    if allowed_features is None:
        return scored_rows
    restricted: list[dict[str, str]] = []
    for row in scored_rows:
        updated = dict(row)
        if row["feature_name"] not in allowed_features:
            updated["score"] = f"{outside_score:.6f}"
        restricted.append(updated)
    return restricted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--restrict-to-config",
        help="Optional config whose causal mask defines the only features allowed to retain learned scores.",
    )
    parser.add_argument(
        "--outside-score",
        type=float,
        default=-1.0,
        help="Score assigned to features outside the allowed support when --restrict-to-config is used.",
    )
    args = parser.parse_args()

    payload = torch.load(Path(args.model), map_location="cpu")
    feature_columns = list(payload["feature_columns"])
    hidden_dim = int(payload.get("hidden_dim", 32))
    model = build_discovery_model(len(feature_columns), hidden_dim=hidden_dim)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    bundle = load_dataset(load_config(Path(args.config)))
    rows = build_feature_clue_rows(bundle, split_name=args.split)
    x = torch.tensor([[float(row[key]) for key in feature_columns] for row in rows], dtype=torch.float32)
    with torch.no_grad():
        rank_logits, support_logits = model(x)
        scores = combine_discovery_scores(rank_logits, support_logits).squeeze(1).tolist()
        support_scores = torch.sigmoid(support_logits).squeeze(1).tolist()
        rank_scores = torch.sigmoid(rank_logits).squeeze(1).tolist()
    scored_rows = []
    for row, score, support_score, rank_score in zip(rows, scores, support_scores, rank_scores, strict=True):
        scored_rows.append(
            {
                "dataset": row["dataset"],
                "feature_index": row["feature_index"],
                "feature_name": row["feature_name"],
                "support_score": f"{support_score:.6f}",
                "rank_score": f"{rank_score:.6f}",
                "score": f"{score:.6f}",
            }
        )
    scored_rows = _apply_support_restriction(
        scored_rows,
        _allowed_feature_names(args.restrict_to_config),
        args.outside_score,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scored_rows)
    print(out_path)


if __name__ == "__main__":
    main()
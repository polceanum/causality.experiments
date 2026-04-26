from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.discovery import (
    DISCOVERY_FEATURE_COLUMNS,
    aggregate_rank_target,
    build_discovery_model,
    clue_feature_vector,
    combine_discovery_scores,
)


def _pairwise_ranking_loss(logits: torch.Tensor, labels: torch.Tensor, dataset_ids: list[str], margin: float) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for dataset_id in sorted(set(dataset_ids)):
        idxs = [index for index, value in enumerate(dataset_ids) if value == dataset_id]
        if len(idxs) < 2:
            continue
        dataset_logits = logits[idxs, 0]
        dataset_labels = labels[idxs, 0]
        pos = dataset_logits[dataset_labels > 0.5]
        neg = dataset_logits[dataset_labels <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        losses.append(torch.relu(margin - (pos[:, None] - neg[None, :])).mean())
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = float(text)
    if not torch.isfinite(torch.tensor(parsed)):
        return None
    return parsed


def _has_utility_supervision(row: dict[str, str]) -> bool:
    utility_target = _parse_optional_float(row.get("utility_target"))
    utility_weight = _parse_optional_float(row.get("utility_weight"))
    return utility_target is not None and (utility_weight or 0.0) > 0.0


def _row_rank_target(row: dict[str, str], utility_blend: float) -> float:
    prepared: dict[str, float | bool] = {}
    for key in (
        "has_explicit_supervision",
        "has_cause_position",
    ):
        prepared[key] = str(row.get(key, "")).lower() == "true"
    for key in (
        "causal_target",
        "cause_position_target",
        "corr_margin",
        "utility_target",
        "utility_weight",
    ):
        parsed = _parse_optional_float(row.get(key))
        prepared[key] = float("nan") if parsed is None else parsed
    return aggregate_rank_target(prepared, utility_blend=utility_blend)


def _utility_loss(rank_logits: torch.Tensor, rows: list[dict[str, str]]) -> torch.Tensor:
    targets = []
    weights = []
    indices = []
    for index, row in enumerate(rows):
        utility_target = _parse_optional_float(row.get("utility_target"))
        utility_weight = _parse_optional_float(row.get("utility_weight"))
        if utility_target is None or utility_weight is None or utility_weight <= 0.0:
            continue
        indices.append(index)
        targets.append(utility_target)
        weights.append(utility_weight)
    if not indices:
        return rank_logits.new_tensor(0.0)
    target_tensor = torch.tensor(targets, dtype=rank_logits.dtype, device=rank_logits.device).unsqueeze(1)
    weight_tensor = torch.tensor(weights, dtype=rank_logits.dtype, device=rank_logits.device).unsqueeze(1)
    logits = rank_logits[indices]
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, target_tensor, reduction="none")
    return (losses * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Discovery dataset CSV.")
    parser.add_argument("--out", required=True, help="Output model path.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--ranking-margin", type=float, default=0.2)
    parser.add_argument("--support-weight", type=float, default=0.5)
    parser.add_argument("--utility-loss-weight", type=float, default=0.5)
    parser.add_argument("--utility-target-blend", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--exclude-dataset", action="append", default=[], help="Dataset name to exclude from discovery supervision.")
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.data).open("r", encoding="utf-8", newline="")))
    excluded = {name.strip() for name in args.exclude_dataset if name.strip()}
    train_rows = [
        row
        for row in rows
        if row.get("dataset", "") not in excluded
        and (
            row.get("has_explicit_supervision", "").lower() == "true"
            or _has_utility_supervision(row)
        )
    ]
    if not train_rows:
        raise ValueError("No explicit discovery supervision rows remain after filtering.")
    x = torch.tensor(
        [clue_feature_vector(row, DISCOVERY_FEATURE_COLUMNS) for row in train_rows],
        dtype=torch.float32,
    )
    y = torch.tensor(
        [_row_rank_target(row, utility_blend=args.utility_target_blend) for row in train_rows],
        dtype=torch.float32,
    ).unsqueeze(1)
    support_y = torch.tensor(
        [
            float(row["causal_target"])
            if row.get("has_explicit_supervision", "").lower() == "true"
            else float(row.get("soft_causal_target", 0.0))
            for row in train_rows
        ],
        dtype=torch.float32,
    ).unsqueeze(1)
    dataset_ids = [str(row["dataset"]) for row in train_rows]

    model = build_discovery_model(len(DISCOVERY_FEATURE_COLUMNS), hidden_dim=args.hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        rank_logits, support_logits = model(x)
        scores = combine_discovery_scores(rank_logits, support_logits)
        pointwise_loss = torch.nn.functional.binary_cross_entropy_with_logits(rank_logits, y)
        support_loss = torch.nn.functional.binary_cross_entropy_with_logits(support_logits, support_y)
        ranking_loss = _pairwise_ranking_loss(scores, support_y, dataset_ids, margin=args.ranking_margin)
        utility_loss = _utility_loss(rank_logits, train_rows)
        loss = (
            pointwise_loss
            + args.support_weight * support_loss
            + args.ranking_weight * ranking_loss
            + args.utility_loss_weight * utility_loss
        )
        loss.backward()
        opt.step()

    with torch.no_grad():
        rank_logits, support_logits = model(x)
        probs = combine_discovery_scores(rank_logits, support_logits)
        pointwise_loss = torch.nn.functional.binary_cross_entropy_with_logits(rank_logits, y).item()
        support_loss = torch.nn.functional.binary_cross_entropy_with_logits(support_logits, support_y).item()
        ranking_loss = _pairwise_ranking_loss(probs, support_y, dataset_ids, margin=args.ranking_margin).item()
        utility_loss = _utility_loss(rank_logits, train_rows).item()
        accuracy = float(((probs >= 0.5) == (support_y >= 0.5)).float().mean().item())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": list(DISCOVERY_FEATURE_COLUMNS),
            "hidden_dim": args.hidden_dim,
            "train_summary": {
                "rows": len(train_rows),
                "datasets": sorted(set(dataset_ids)),
                "excluded_datasets": sorted(excluded),
                "pointwise_loss": pointwise_loss,
                "support_loss": support_loss,
                "ranking_loss": ranking_loss,
                "utility_loss": utility_loss,
                "train_accuracy": accuracy,
                "ranking_margin": args.ranking_margin,
                "ranking_weight": args.ranking_weight,
                "support_weight": args.support_weight,
                "utility_loss_weight": args.utility_loss_weight,
                "utility_target_blend": args.utility_target_blend,
            },
        },
        out_path,
    )
    print(out_path)


if __name__ == "__main__":
    main()
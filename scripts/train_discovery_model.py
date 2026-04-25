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
    build_discovery_model,
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
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--exclude-dataset", action="append", default=[], help="Dataset name to exclude from discovery supervision.")
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.data).open("r", encoding="utf-8", newline="")))
    excluded = {name.strip() for name in args.exclude_dataset if name.strip()}
    train_rows = [
        row
        for row in rows
        if row.get("has_explicit_supervision", "").lower() == "true" and row.get("dataset", "") not in excluded
    ]
    if not train_rows:
        raise ValueError("No explicit discovery supervision rows remain after filtering.")
    x = torch.tensor(
        [[float(row[key]) for key in DISCOVERY_FEATURE_COLUMNS] for row in train_rows],
        dtype=torch.float32,
    )
    y = torch.tensor([float(row["soft_causal_target"]) for row in train_rows], dtype=torch.float32).unsqueeze(1)
    support_y = torch.tensor([float(row["causal_target"]) for row in train_rows], dtype=torch.float32).unsqueeze(1)
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
        loss = pointwise_loss + args.support_weight * support_loss + args.ranking_weight * ranking_loss
        loss.backward()
        opt.step()

    with torch.no_grad():
        rank_logits, support_logits = model(x)
        probs = combine_discovery_scores(rank_logits, support_logits)
        pointwise_loss = torch.nn.functional.binary_cross_entropy_with_logits(rank_logits, y).item()
        support_loss = torch.nn.functional.binary_cross_entropy_with_logits(support_logits, support_y).item()
        ranking_loss = _pairwise_ranking_loss(probs, support_y, dataset_ids, margin=args.ranking_margin).item()
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
                "train_accuracy": accuracy,
                "ranking_margin": args.ranking_margin,
                "ranking_weight": args.ranking_weight,
                "support_weight": args.support_weight,
            },
        },
        out_path,
    )
    print(out_path)


if __name__ == "__main__":
    main()
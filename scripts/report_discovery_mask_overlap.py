from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.config import load_config
from causality_experiments.data import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Base config used to derive the heuristic mask.")
    parser.add_argument("--scores", required=True, help="CSV of discovery scores with feature_name and score columns.")
    parser.add_argument("--top-k", type=int, default=512, help="Number of learned features to compare.")
    parser.add_argument("--out", help="Optional CSV output path.")
    args = parser.parse_args()

    bundle = load_dataset(load_config(Path(args.config)))
    if bundle.causal_mask is None:
        raise ValueError("Base config did not produce a heuristic causal mask.")
    feature_names = list((bundle.metadata or {}).get("feature_columns", []))
    heuristic = {
        feature_name
        for feature_name, keep in zip(feature_names, bundle.causal_mask.tolist(), strict=True)
        if keep > 0.0
    }

    scored = []
    with Path(args.scores).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            feature_name = str(row.get("feature_name", "")).strip()
            if feature_name:
                scored.append((feature_name, float(row["score"])))
    learned = {name for name, _ in sorted(scored, key=lambda item: item[1], reverse=True)[: args.top_k]}

    overlap = heuristic & learned
    precision = len(overlap) / max(len(learned), 1)
    recall = len(overlap) / max(len(heuristic), 1)
    jaccard = len(overlap) / max(len(heuristic | learned), 1)
    row = {
        "config": Path(args.config).stem,
        "heuristic_size": len(heuristic),
        "learned_size": len(learned),
        "overlap_size": len(overlap),
        "precision_vs_heuristic": f"{precision:.3f}",
        "recall_vs_heuristic": f"{recall:.3f}",
        "jaccard": f"{jaccard:.3f}",
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

    print(",".join(row.keys()))
    print(",".join(str(value) for value in row.values()))


if __name__ == "__main__":
    main()
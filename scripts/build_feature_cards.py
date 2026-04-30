from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.clues import build_feature_cards, write_csv_rows
from causality_experiments.config import load_config
from causality_experiments.data import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True, help="Config used to build feature cards. Can be passed multiple times.")
    parser.add_argument("--split", default="train", help="Split used to summarize feature activations.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of top and bottom activations to summarize per feature.")
    parser.add_argument("--out", required=True, help="CSV path for feature-card rows.")
    args = parser.parse_args()

    rows = []
    for config_path in args.config:
        config = load_config(Path(config_path))
        bundle = load_dataset(config)
        rows.extend(build_feature_cards(bundle, split_name=args.split, top_k=args.top_k))
    out_path = Path(args.out)
    write_csv_rows(out_path, rows)
    print(out_path)


if __name__ == "__main__":
    main()

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
from causality_experiments.discovery import build_feature_clue_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True, help="Config used to build the dataset bundle. Can be passed multiple times.")
    parser.add_argument("--split", default="train", help="Split used to generate clue rows.")
    parser.add_argument("--out", required=True, help="CSV path for the discovery dataset rows.")
    args = parser.parse_args()

    rows = []
    for config_path in args.config:
        config = load_config(Path(config_path))
        bundle = load_dataset(config)
        rows.extend(build_feature_clue_rows(bundle, split_name=args.split))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)


if __name__ == "__main__":
    main()
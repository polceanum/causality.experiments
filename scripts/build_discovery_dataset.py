from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.clues import read_csv_rows, write_csv_rows
from causality_experiments.config import load_config
from causality_experiments.data import load_dataset
from causality_experiments.discovery import build_feature_clue_rows, merge_external_clue_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True, help="Config used to build the dataset bundle. Can be passed multiple times.")
    parser.add_argument("--split", default="train", help="Split used to generate clue rows.")
    parser.add_argument("--clues", action="append", default=[], help="External clue CSV to merge by dataset and feature_name. Can be passed multiple times.")
    parser.add_argument("--out", required=True, help="CSV path for the discovery dataset rows.")
    args = parser.parse_args()

    external_clues = []
    for clue_path in args.clues:
        external_clues.extend(read_csv_rows(Path(clue_path)))

    rows = []
    for config_path in args.config:
        config = load_config(Path(config_path))
        bundle = load_dataset(config)
        clue_rows = build_feature_clue_rows(bundle, split_name=args.split)
        if external_clues:
            clue_rows = merge_external_clue_rows(clue_rows, external_clues)
        rows.extend(clue_rows)
    out_path = Path(args.out)
    write_csv_rows(out_path, rows)
    print(out_path)


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.clues import build_language_clue_rows, read_csv_rows, write_csv_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", action="append", required=True, help="Feature-card CSV. Can be passed multiple times.")
    parser.add_argument("--domain", default="auto", help="Language template domain: auto, waterbirds, or general.")
    parser.add_argument("--out", required=True, help="CSV path for language clue rows.")
    args = parser.parse_args()

    cards = []
    for card_path in args.cards:
        cards.extend(read_csv_rows(Path(card_path)))
    rows = build_language_clue_rows(cards, domain=args.domain)
    out_path = Path(args.out)
    write_csv_rows(out_path, rows)
    print(out_path)


if __name__ == "__main__":
    main()

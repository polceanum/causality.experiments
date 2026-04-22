from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.run import run_experiment


def main() -> None:
    for config in sorted(Path("configs/experiments").glob("*.yaml")):
        run_dir = run_experiment(config)
        print(run_dir)


if __name__ == "__main__":
    main()

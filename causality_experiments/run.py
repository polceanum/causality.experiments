from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import load_config
from .data import load_dataset
from .literature import benchmark_metadata
from .methods import fit_method
from .metrics import evaluate


def run_experiment(config_path: str | Path, output_root: str | Path | None = None) -> Path:
    config = load_config(config_path)
    bundle = load_dataset(config)
    model = fit_method(bundle, config)
    metrics = evaluate(model, bundle, config)
    root = Path(output_root or config.get("output_dir", "outputs/runs"))
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{config['name']}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": config,
        "benchmark": benchmark_metadata(config),
        "dataset": {
            "name": bundle.name,
            "input_dim": bundle.input_dim,
            "output_dim": bundle.output_dim,
            "metadata": bundle.metadata or {},
        },
        "metrics": metrics,
    }
    model_details = getattr(model, "details", None)
    if model_details is not None:
        payload["model_details"] = model_details
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in sorted(metrics.items()):
            writer.writerow([key, value])
    _plot_metrics(metrics, run_dir / "summary.png")
    return run_dir


def _plot_metrics(metrics: dict[str, float], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    acc_items = [(k, v) for k, v in metrics.items() if k.endswith("accuracy")]
    if not acc_items:
        return
    labels = [k.replace("/", "\n") for k, _ in acc_items]
    values = [v for _, v in acc_items]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4))
    ax.bar(labels, values, color="#3f6f8f")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Accuracy summary")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summarize_runs(runs_dir: str | Path) -> Path:
    root = Path(runs_dir)
    rows: list[dict[str, str]] = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        row = {
            "run": metrics_path.parent.name,
            "dataset": payload.get("dataset", {}).get("name", ""),
            "benchmark_id": payload.get("benchmark", {}).get("id", ""),
            "benchmark_kind": payload.get("benchmark", {}).get("kind", ""),
            "literature_comparable": str(
                payload.get("benchmark", {}).get("comparable_to_literature", False)
            ),
            "benchmark_provenance_complete": str(
                payload.get("benchmark", {}).get("provenance_complete", False)
            ),
            "benchmark_feature_extractor": str(
                payload.get("benchmark", {}).get("provenance", {}).get("feature_extractor", "")
            ),
            "benchmark_feature_source": str(
                payload.get("benchmark", {}).get("provenance", {}).get("feature_source", "")
            ),
            "benchmark_split_definition": str(
                payload.get("benchmark", {}).get("provenance", {}).get("split_definition", "")
            ),
            "config": config.get("name", ""),
            "method": config.get("method", {}).get("kind", ""),
        }
        row.update({k: str(v) for k, v in payload.get("metrics", {}).items()})
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {root}.")
    keys = sorted({key for row in rows for key in row})
    out = root / "summary.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return out

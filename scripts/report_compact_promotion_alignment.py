from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.run import summarize_runs


PAIR_SPECS = [
    {
        "label": "fixed_discovery_top128",
        "compact_config": "waterbirds_features_counterfactual_adversarial_schedule_gated_nuisance0p9_compact_fixed_discovery_top128_discovery_full_top128_compact",
        "full_config": "waterbirds_features_counterfactual_adversarial_schedule_gated_nuisance0p9_discovery_full_top128",
    },
    {
        "label": "fixed_instability_jtt_top128",
        "compact_config": "waterbirds_features_counterfactual_adversarial_schedule_fixed_instability_jtt_gated_nuisance0p9_compact_fixed_instability_jtt_discovery_top128_discovery_full_top128_compact",
        "full_config": "waterbirds_features_counterfactual_adversarial_schedule_fixed_instability_jtt_gated_nuisance0p9_discovery_full_top128",
    },
    {
        "label": "grouped_discovery_top128",
        "compact_config": "waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_gate_nuisance0p9_compact_grouped_discovery_top128_discovery_full_top128_compact",
        "full_config": "waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_gate_nuisance0p9_discovery_full_top128",
    },
    {
        "label": "grouped_instability_jtt_top128",
        "compact_config": "waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_instability_jtt_gate_nuisance0p9_compact_grouped_instability_jtt_discovery_top128_discovery_full_top128_compact",
        "full_config": "waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_instability_jtt_gate_nuisance0p9_discovery_full_top128",
    },
    {
        "label": "grouped_instability_jtt_sweepwinner_top128",
        "compact_config": "waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_instability_jtt_gate_nuisance0p9_discovery_full_top128_stage1e20_topf0p15_upw3p0_compact",
        "full_config": "waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_instability_jtt_gate_nuisance0p9_discovery_full_top128_stage1e20_topf0p15_upw3p0",
        "compact_csv": "outputs/runs/waterbirds-instability-jtt-sweep.csv",
        "compact_csv_config": "waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_instability_jtt_gate_nuisance0p9_discovery_full_top128_stage1e20_topf0p15_upw3p0",
    },
]


def _safe_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="outputs/runs")
    parser.add_argument(
        "--out",
        default="outputs/runs/waterbirds-compact-promotion-alignment.csv",
    )
    args = parser.parse_args()

    summary = summarize_runs(args.runs)
    rows = list(csv.DictReader(summary.open()))
    latest_by_config: dict[str, dict[str, str]] = {}
    for row in rows:
        config = row.get("config", "")
        if not config:
            continue
        if config not in latest_by_config or row.get("run", "") > latest_by_config[config].get("run", ""):
            latest_by_config[config] = row

    compact_csv_rows: dict[tuple[str, str], dict[str, str]] = {}
    for spec in PAIR_SPECS:
        compact_csv = spec.get("compact_csv")
        if not compact_csv:
            continue
        compact_path = Path(str(compact_csv))
        if not compact_path.exists():
            continue
        for row in csv.DictReader(compact_path.open()):
            key = (str(spec["label"]), row.get("config", ""))
            compact_csv_rows[key] = row

    output_rows: list[dict[str, str]] = []
    for spec in PAIR_SPECS:
        compact = latest_by_config.get(spec["compact_config"], {})
        compact_csv_key = str(spec.get("compact_csv_config", spec["compact_config"]))
        compact_csv_row = compact_csv_rows.get((str(spec["label"]), compact_csv_key), {})
        full = latest_by_config.get(spec["full_config"], {})
        compact_wga = _safe_float(compact.get("test/worst_group_accuracy"))
        compact_val_wga = _safe_float(compact.get("val/worst_group_accuracy"))
        compact_run = compact.get("run", "")
        if compact_csv_row:
            compact_wga = _safe_float(compact_csv_row.get("test_wga"))
            compact_val_wga = _safe_float(compact_csv_row.get("val_wga"))
            compact_run = compact_csv_row.get("run", "")
        full_wga = _safe_float(full.get("test/worst_group_accuracy"))
        output_rows.append(
            {
                "label": str(spec["label"]),
                "compact_run": compact_run,
                "compact_test_wga": "" if compact_wga is None else f"{compact_wga:.6f}",
                "compact_val_wga": "" if compact_val_wga is None else f"{compact_val_wga:.6f}",
                "full_run": full.get("run", ""),
                "full_test_wga": "" if full_wga is None else f"{full_wga:.6f}",
                "full_val_wga": "" if _safe_float(full.get("val/worst_group_accuracy")) is None else f"{_safe_float(full.get('val/worst_group_accuracy')):.6f}",
                "test_wga_gap_full_minus_compact": "" if compact_wga is None or full_wga is None else f"{full_wga - compact_wga:.6f}",
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)
    print(out_path)
    for row in output_rows:
        print(row)


if __name__ == "__main__":
    main()
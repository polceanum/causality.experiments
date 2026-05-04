from __future__ import annotations

import json
from pathlib import Path

from scripts.package_waterbirds_bridge_incumbent import build_package


def test_build_package_collects_bridge_claim_and_artifacts(tmp_path: Path) -> None:
    candidate_report = tmp_path / "candidate.json"
    candidate_report.write_text(
        json.dumps(
            {
                "rows": {
                    "candidate": {
                        "mean_test_wga": 0.94,
                        "min_test_wga": 0.93,
                        "mean_delta_to_baseline": 0.01,
                        "mean_delta_to_stats": 0.005,
                    }
                },
                "candidate_vs_best_random": {
                    "mean_delta_to_best_random": 0.002,
                    "min_delta_to_best_random": 0.0,
                    "non_negative_best_random_seeds": 5,
                    "paired": [],
                },
            }
        ),
        encoding="utf-8",
    )
    support_report = tmp_path / "support.json"
    support_report.write_text(
        json.dumps(
            {
                "top_k": 512,
                "reference_label": "stats",
                "supports": [
                    {
                        "label": "stats",
                        "env_ge_label_count": 90,
                    },
                    {
                        "label": "bridge_fused_w0p3",
                        "overlap_with_reference": 311,
                        "jaccard_with_reference": 0.436,
                        "env_ge_label_count": 5,
                    },
                    {"label": "random_score_0", "env_ge_label_count": 91},
                    {"label": "random_score_1", "env_ge_label_count": 93},
                ],
            }
        ),
        encoding="utf-8",
    )
    critical_audit = tmp_path / "audit.json"
    critical_audit.write_text(json.dumps({"critical_read": ["keep incumbent"], "incumbent_seed_rows": [{"seed": 101}]}), encoding="utf-8")
    runner_csv = tmp_path / "runner.csv"
    runner_csv.write_text("seed,label\n101,bridge\n", encoding="utf-8")
    runner_json = tmp_path / "runner.json"
    runner_json.write_text("{}", encoding="utf-8")
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    (score_dir / "scores_bridge.csv").write_text("feature,score\n", encoding="utf-8")
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "manifest.json").write_text("{}", encoding="utf-8")

    package = build_package(
        candidate_report_path=candidate_report,
        support_report_path=support_report,
        critical_audit_path=critical_audit,
        runner_csv_path=runner_csv,
        runner_json_path=runner_json,
        score_dir=score_dir,
        trace_dir=trace_dir,
        candidate_label="bridge_fused_w0p3_top512",
    )

    assert package["headline"]["mean_test_wga"] == 0.94
    assert package["support_summary"]["bridge_overlap_with_stats"] == 311
    assert package["support_summary"]["random_env_ge_label_count_max"] == 93
    assert package["artifacts"]["runner_csv"]["sha256"]
    assert package["seed_rows"] == [{"seed": 101}]

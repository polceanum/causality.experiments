from pathlib import Path

from scripts.report_waterbirds_bridge_candidate import build_report


def test_build_report_recomputes_best_random_gate(tmp_path: Path) -> None:
    input_csv = tmp_path / "rows.csv"
    input_csv.write_text(
        "seed,top_k,label,row_type,test_wga,delta_to_baseline,delta_to_stats\n"
        "101,512,official_dfr,baseline,0.93,,\n"
        "101,512,stats_top512,stats_control,0.94,0.01,\n"
        "101,512,random_score_0_top512,random_control,0.91,-0.02,-0.03\n"
        "101,512,random_score_1_top512,random_control,0.935,0.005,-0.005\n"
        "101,512,bridge_fused_w0p3_top512,candidate,0.945,0.015,0.005\n"
        "102,512,official_dfr,baseline,0.92,,\n"
        "102,512,stats_top512,stats_control,0.925,0.005,\n"
        "102,512,random_score_0_top512,random_control,0.93,0.01,0.005\n"
        "102,512,random_score_1_top512,random_control,0.91,-0.01,-0.015\n"
        "102,512,bridge_fused_w0p3_top512,candidate,0.93,0.01,0.005\n",
        encoding="utf-8",
    )
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    (score_dir / "scores_stats.csv").write_text("feature_name,score\nf0,0.1\n", encoding="utf-8")
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "manifest.json").write_text("{}", encoding="utf-8")

    report = build_report(
        input_csv=input_csv,
        input_json=None,
        score_dir=score_dir,
        trace_dir=trace_dir,
        candidate_label="bridge_fused_w0p3_top512",
    )

    assert report["rows"]["candidate"]["count"] == 2
    assert report["rows"]["candidate"]["mean_test_wga"] == 0.9375
    assert report["candidate_vs_best_random"]["non_negative_best_random_seeds"] == 2
    assert report["candidate_vs_best_random"]["min_delta_to_best_random"] == 0.0
    assert report["trace_manifest"]["manifest_count"] == 1
    assert report["score_files"][0]["path"].endswith("scores_stats.csv")

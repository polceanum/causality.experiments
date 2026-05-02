from pathlib import Path

from scripts.report_waterbirds_bridge_support import build_support_report


def test_build_support_report_compares_score_supports(tmp_path: Path) -> None:
    clues = tmp_path / "clues.csv"
    clues.write_text(
        "dataset,feature_name,label_corr,env_corr,corr_margin\n"
        "waterbirds_features,f0,0.9,0.1,0.8\n"
        "waterbirds_features,f1,0.7,0.2,0.5\n"
        "waterbirds_features,f2,0.2,0.8,-0.6\n",
        encoding="utf-8",
    )
    stats = tmp_path / "scores_stats.csv"
    stats.write_text(
        "dataset,feature_index,feature_name,score\n"
        "waterbirds_features,0,f0,0.9\n"
        "waterbirds_features,1,f1,0.8\n"
        "waterbirds_features,2,f2,0.1\n",
        encoding="utf-8",
    )
    bridge = tmp_path / "scores_bridge.csv"
    bridge.write_text(
        "dataset,feature_index,feature_name,score\n"
        "waterbirds_features,2,f2,0.95\n"
        "waterbirds_features,0,f0,0.7\n"
        "waterbirds_features,1,f1,0.2\n",
        encoding="utf-8",
    )

    report = build_support_report(
        clue_path=clues,
        score_paths={"stats": stats, "bridge": bridge},
        top_k=2,
        reference_label="stats",
    )

    support_by_label = {row["label"]: row for row in report["supports"]}
    assert support_by_label["stats"]["overlap_with_reference"] == 2
    assert support_by_label["bridge"]["overlap_with_reference"] == 1
    assert support_by_label["bridge"]["env_ge_label_count"] == 1
    assert support_by_label["bridge"]["only_vs_reference_count"] == 1

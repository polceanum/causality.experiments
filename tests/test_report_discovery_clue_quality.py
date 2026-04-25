from pathlib import Path

from scripts.report_discovery_clue_quality import summarize_topk_clue_quality


def test_summarize_topk_clue_quality_prefers_high_scoring_rows(tmp_path: Path) -> None:
    clue_path = tmp_path / "clues.csv"
    clue_path.write_text(
        "dataset,feature_name,label_corr,env_corr,corr_margin,causal_target,supervision_explicit\n"
        "waterbirds_features,feature_0,0.9,0.1,0.8,1.0,0.0\n"
        "waterbirds_features,feature_1,0.4,0.3,0.1,0.0,0.0\n"
        "waterbirds_features,feature_2,0.2,0.7,-0.5,0.0,0.0\n",
        encoding="utf-8",
    )
    score_path = tmp_path / "scores.csv"
    score_path.write_text(
        "dataset,feature_index,feature_name,score\n"
        "waterbirds_features,0,feature_0,0.95\n"
        "waterbirds_features,1,feature_1,0.50\n"
        "waterbirds_features,2,feature_2,0.10\n",
        encoding="utf-8",
    )

    row = summarize_topk_clue_quality(clue_path, score_path, top_k=2, label="learned")
    assert row["label"] == "learned"
    assert row["top_k"] == "2"
    assert float(row["mean_label_corr"]) > float(row["mean_env_corr"])
    assert float(row["mean_corr_margin"]) > 0.0
    assert float(row["mean_causal_target"]) == 0.5


def test_summarize_topk_clue_quality_requires_overlap(tmp_path: Path) -> None:
    clue_path = tmp_path / "clues.csv"
    clue_path.write_text(
        "dataset,feature_name,label_corr,env_corr,corr_margin,causal_target,supervision_explicit\n"
        "waterbirds_features,feature_0,0.9,0.1,0.8,1.0,0.0\n",
        encoding="utf-8",
    )
    score_path = tmp_path / "scores.csv"
    score_path.write_text(
        "dataset,feature_index,feature_name,score\n"
        "waterbirds_features,feature_9,0,0.95\n",
        encoding="utf-8",
    )

    try:
        summarize_topk_clue_quality(clue_path, score_path, top_k=1, label="random")
    except ValueError as exc:
        assert "No overlapping features" in str(exc)
    else:
        raise AssertionError("Expected summarize_topk_clue_quality to reject non-overlapping score rows.")
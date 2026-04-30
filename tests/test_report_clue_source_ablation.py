from pathlib import Path

from scripts.report_clue_source_ablation import summarize_source_ablation


def test_summarize_source_ablation_reports_language_and_reference_overlap(tmp_path: Path) -> None:
    clue_path = tmp_path / "clues.csv"
    clue_path.write_text(
        "dataset,feature_name,label_corr,env_corr,corr_margin,language_causal_score,language_spurious_score,language_confidence,top_activation_group_entropy,label_env_disentanglement\n"
        "waterbirds_features,feature_0,0.9,0.1,0.8,1.0,0.0,1.0,0.5,0.8\n"
        "waterbirds_features,feature_1,0.4,0.3,0.1,0.5,0.5,0.0,1.0,0.1\n"
        "waterbirds_features,feature_2,0.2,0.7,-0.5,0.0,1.0,1.0,0.2,0.5\n",
        encoding="utf-8",
    )
    stats_path = tmp_path / "stats.csv"
    stats_path.write_text(
        "feature_name,score\nfeature_0,0.9\nfeature_1,0.8\nfeature_2,0.1\n",
        encoding="utf-8",
    )
    language_path = tmp_path / "language.csv"
    language_path.write_text(
        "feature_name,score\nfeature_0,0.95\nfeature_2,0.6\nfeature_1,0.5\n",
        encoding="utf-8",
    )

    rows = summarize_source_ablation(
        clue_path,
        [("stats", stats_path), ("language", language_path)],
        top_k_values=[2],
        reference_label="stats",
    )

    by_label = {row["label"]: row for row in rows}
    assert by_label["stats"]["top_k"] == "2"
    assert float(by_label["language"]["mean_language_confidence"]) > 0.0
    assert by_label["language"]["overlap_reference"] == "1"
    assert 0.0 < float(by_label["language"]["jaccard_reference"]) < 1.0


def test_summarize_source_ablation_rejects_empty_overlap(tmp_path: Path) -> None:
    clue_path = tmp_path / "clues.csv"
    clue_path.write_text("feature_name,label_corr\nfeature_0,0.5\n", encoding="utf-8")
    score_path = tmp_path / "scores.csv"
    score_path.write_text("feature_name,score\nfeature_9,0.9\n", encoding="utf-8")

    try:
        summarize_source_ablation(clue_path, [("missing", score_path)], top_k_values=[1])
    except ValueError as exc:
        assert "No overlapping features" in str(exc)
    else:
        raise AssertionError("Expected source ablation to reject non-overlapping score rows.")

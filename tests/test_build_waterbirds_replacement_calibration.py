from pathlib import Path

from scripts.build_waterbirds_replacement_calibration import build_calibration_rows, write_csv


def test_build_calibration_rows_joins_support_and_outcomes(tmp_path: Path) -> None:
    clues = tmp_path / "clues.csv"
    clues.write_text(
        "feature_name,label_corr,env_corr,corr_margin\n"
        "f0,0.9,0.1,0.8\n"
        "f1,0.8,0.2,0.6\n"
        "f2,0.2,0.9,-0.7\n",
        encoding="utf-8",
    )
    reference = tmp_path / "reference.csv"
    reference.write_text(
        "feature_name,score\n"
        "f0,1.0\n"
        "f1,0.9\n"
        "f2,0.1\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.csv"
    candidate.write_text(
        "feature_name,score,active_boundary_pair_role,active_boundary_pair_delta\n"
        "f0,1.0,unchanged,0.0\n"
        "f2,0.95,accepted,0.02\n"
        "f1,0.1,evicted,-0.02\n",
        encoding="utf-8",
    )
    outcome = tmp_path / "outcome.csv"
    outcome.write_text(
        "seed,top_k,label,row_type,test_wga,delta_to_baseline,delta_to_stats\n"
        "101,2,random_score_0_top2,random_control,0.91,-0.01,-0.02\n"
        "101,2,variant_top2,candidate,0.94,0.02,0.01\n"
        "102,2,random_score_0_top2,random_control,0.93,0.00,-0.01\n"
        "102,2,variant_top2,candidate,0.935,0.01,-0.005\n",
        encoding="utf-8",
    )

    rows = build_calibration_rows(
        clue_path=clues,
        reference_score_path=reference,
        candidates={"variant_top2": (candidate, outcome)},
        top_k=2,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["changed_count"] == 1
    assert row["overlap_with_reference"] == 1
    assert row["accepted_pair_count"] == 1
    assert row["evicted_pair_count"] == 1
    assert row["candidate_env_ge_label_count"] == 1.0
    assert row["entered_env_ge_label_count"] == 1.0
    assert row["left_env_ge_label_count"] == 0.0
    assert row["outcome_count"] == 2
    assert row["mean_delta_to_stats"] == 0.0025
    assert row["non_negative_stats_seeds"] == 1
    assert row["clears_baseline_gate"] == 1
    assert row["clears_stats_gate"] == 0
    assert row["clears_best_random_gate"] == 1


def test_write_csv_handles_empty_optional_columns(tmp_path: Path) -> None:
    output = tmp_path / "calibration.csv"

    write_csv(output, [{"label": "a", "changed_count": 1}, {"label": "b", "mean_test_wga": 0.9}])

    text = output.read_text(encoding="utf-8")
    assert "changed_count" in text
    assert "mean_test_wga" in text
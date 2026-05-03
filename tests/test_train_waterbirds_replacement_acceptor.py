from scripts.train_waterbirds_replacement_acceptor import build_acceptor_report


def _row(label: str, *, changed: float, target: float, stats_gate: int = 1, random_gate: int = 1) -> dict[str, str]:
    return {
        "label": label,
        "changed_fraction": str(changed / 512.0),
        "jaccard_with_reference": str(1.0 - changed / 512.0),
        "candidate_env_ge_label_count": "5",
        "entered_count": str(changed),
        "entered_env_ge_label_count": "0",
        "entered_mean_label_corr": "0.7",
        "entered_mean_env_corr": "0.1",
        "entered_mean_corr_margin": "0.6",
        "left_count": str(changed),
        "left_env_ge_label_count": "0",
        "left_mean_label_corr": "0.6",
        "left_mean_env_corr": "0.1",
        "left_mean_corr_margin": "0.5",
        "accepted_pair_count": "0",
        "mean_accepted_pair_delta": "0",
        "max_accepted_pair_delta": "0",
        "changed_count": str(changed),
        "mean_delta_to_stats": str(target),
        "mean_delta_to_best_random": str(target),
        "clears_stats_gate": str(stats_gate),
        "clears_best_random_gate": str(random_gate),
        "outcome_count": "2",
    }


def test_acceptor_recommends_positive_gated_rows_when_uncertainty_allows() -> None:
    rows = [
        _row("small_positive", changed=5, target=0.01),
        _row("medium_negative", changed=80, target=-0.01, stats_gate=0, random_gate=0),
        _row("large_negative", changed=140, target=-0.02, stats_gate=0, random_gate=0),
    ]

    report = build_acceptor_report(calibration_rows=rows, alpha=1.0, uncertainty_scale=0.0, min_outcome_count=2)

    recommended = {row["label"] for row in report["recommended"]}
    assert recommended == {"small_positive"}


def test_acceptor_uncertainty_vetoes_weak_positive_rows() -> None:
    rows = [
        _row("weak_positive", changed=5, target=0.0001),
        _row("negative_a", changed=80, target=-0.01, stats_gate=0, random_gate=0),
        _row("negative_b", changed=140, target=-0.02, stats_gate=0, random_gate=0),
    ]

    report = build_acceptor_report(calibration_rows=rows, alpha=1.0, uncertainty_scale=2.0, min_outcome_count=2)

    assert report["residual_std"] > 0.0
    assert report["recommended"] == []


def test_acceptor_requires_enough_outcome_rows() -> None:
    rows = [
        _row("compact_positive", changed=5, target=0.01),
        _row("negative_a", changed=80, target=-0.01, stats_gate=0, random_gate=0),
        _row("negative_b", changed=140, target=-0.02, stats_gate=0, random_gate=0),
    ]

    report = build_acceptor_report(calibration_rows=rows, alpha=1.0, uncertainty_scale=0.0, min_outcome_count=5)

    compact = next(row for row in report["rows"] if row["label"] == "compact_positive")
    assert compact["enough_outcomes"] == 0
    assert compact["recommend"] == 0
    assert report["recommended"] == []
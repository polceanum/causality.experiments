from scripts.run_instability_jtt_sweep import _promotion_decision


def test_promotion_decision_requires_both_test_and_val_strength() -> None:
    eligible, score = _promotion_decision(
        test_wga=0.71,
        val_wga=0.70,
        min_test_wga=0.68,
        min_val_wga=0.68,
        max_test_val_gap=0.03,
    )
    assert eligible is True
    assert score == 0.70


def test_promotion_decision_rejects_large_test_val_gap() -> None:
    eligible, score = _promotion_decision(
        test_wga=0.72,
        val_wga=0.66,
        min_test_wga=0.68,
        min_val_wga=0.65,
        max_test_val_gap=0.03,
    )
    assert eligible is False
    assert score == 0.66


def test_promotion_decision_rejects_low_validation_score() -> None:
    eligible, _ = _promotion_decision(
        test_wga=0.74,
        val_wga=0.64,
        min_test_wga=0.68,
        min_val_wga=0.68,
        max_test_val_gap=0.10,
    )
    assert eligible is False
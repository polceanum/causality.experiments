from scripts.run_instability_jtt_sweep import _promotion_decision


def test_compact_gate_check_uses_same_promotion_rule() -> None:
    eligible, score = _promotion_decision(
        test_wga=0.69,
        val_wga=0.68,
        min_test_wga=0.68,
        min_val_wga=0.68,
        max_test_val_gap=0.03,
    )
    row = {
        "promotion_score": score,
        "eligible_for_promotion": int(eligible),
    }
    assert row == {"promotion_score": 0.68, "eligible_for_promotion": 1}
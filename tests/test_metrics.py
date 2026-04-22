from causality_experiments.data import load_dataset
from causality_experiments.methods import fit_method
from causality_experiments.metrics import evaluate


def test_constant_metrics_are_bounded() -> None:
    config = {
        "seed": 3,
        "dataset": {"kind": "synthetic_linear", "n": 120},
        "method": {"kind": "constant"},
    }
    bundle = load_dataset(config)
    model = fit_method(bundle, config)
    metrics = evaluate(model, bundle, config)
    assert 0.0 <= metrics["test/accuracy"] <= 1.0
    assert 0.0 <= metrics["test/worst_group_accuracy"] <= 1.0


def test_oracle_reports_support_recovery() -> None:
    config = {
        "seed": 3,
        "dataset": {"kind": "synthetic_linear", "n": 120},
        "method": {"kind": "oracle"},
    }
    bundle = load_dataset(config)
    model = fit_method(bundle, config)
    metrics = evaluate(model, bundle, config)
    assert metrics["support_recovery"] == 1.0


def test_counterfactual_augmentation_runs() -> None:
    config = {
        "seed": 3,
        "dataset": {"kind": "synthetic_linear", "n": 120},
        "method": {"kind": "counterfactual_augmentation", "hidden_dim": 8},
        "training": {"device": "cpu", "epochs": 1, "batch_size": 32},
    }
    bundle = load_dataset(config)
    model = fit_method(bundle, config)
    metrics = evaluate(model, bundle, config)
    assert 0.0 <= metrics["test/accuracy"] <= 1.0


def test_sequence_counterfactual_augmentation_runs() -> None:
    config = {
        "seed": 3,
        "dataset": {"kind": "text_toy", "n": 120},
        "method": {
            "kind": "counterfactual_augmentation",
            "hidden_dim": 8,
            "embedding_dim": 4,
        },
        "training": {"device": "cpu", "epochs": 1, "batch_size": 32},
    }
    bundle = load_dataset(config)
    model = fit_method(bundle, config)
    metrics = evaluate(model, bundle, config)
    assert 0.0 <= metrics["test/accuracy"] <= 1.0


def test_irm_runs() -> None:
    config = {
        "seed": 3,
        "dataset": {"kind": "synthetic_linear", "n": 120},
        "method": {"kind": "irm", "hidden_dim": 8, "penalty_weight": 1.0},
        "training": {"device": "cpu", "epochs": 1},
    }
    bundle = load_dataset(config)
    model = fit_method(bundle, config)
    metrics = evaluate(model, bundle, config)
    assert 0.0 <= metrics["test/accuracy"] <= 1.0


def test_group_robust_baselines_run() -> None:
    for method in (
        "group_balanced_erm",
        "group_dro",
        "jtt",
        "adversarial_probe",
        "counterfactual_adversarial",
    ):
        config = {
            "seed": 3,
            "dataset": {"kind": "synthetic_linear", "n": 120},
            "method": {"kind": method, "hidden_dim": 8},
            "training": {"device": "cpu", "epochs": 1, "batch_size": 32},
        }
        bundle = load_dataset(config)
        model = fit_method(bundle, config)
        metrics = evaluate(model, bundle, config)
        assert 0.0 <= metrics["test/accuracy"] <= 1.0


def test_probe_diagnostics_are_reported_for_torch_model() -> None:
    config = {
        "seed": 3,
        "dataset": {"kind": "waterbirds_tiny", "n": 120},
        "method": {"kind": "erm", "hidden_dim": 8},
        "training": {"device": "cpu", "epochs": 1, "batch_size": 32},
    }
    bundle = load_dataset(config)
    model = fit_method(bundle, config)
    metrics = evaluate(model, bundle, config)
    assert "probe/causal_accuracy" in metrics
    assert "probe/nuisance_accuracy" in metrics
    assert "probe/selectivity" in metrics


def test_probe_diagnostics_handle_multiclass_environment() -> None:
    config = {
        "seed": 3,
        "dataset": {"kind": "dsprites_tiny", "n": 120},
        "method": {"kind": "erm", "hidden_dim": 8},
        "training": {"device": "cpu", "epochs": 1, "batch_size": 32},
    }
    bundle = load_dataset(config)
    model = fit_method(bundle, config)
    metrics = evaluate(model, bundle, config)
    assert "probe/nuisance_accuracy" in metrics

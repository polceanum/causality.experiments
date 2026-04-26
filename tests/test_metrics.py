from pathlib import Path

from causality_experiments.data import load_dataset
from causality_experiments.methods import FeatureGate, _apply_causal_input_gate, fit_method
from causality_experiments.metrics import evaluate


def test_causal_input_gate_can_downweight_nuisance_features() -> None:
    bundle = load_dataset({"seed": 3, "dataset": {"kind": "synthetic_linear", "n": 120}})
    xb = bundle.split("train")["x"][:1]
    gated = _apply_causal_input_gate(
        bundle,
        xb,
        {"causal_input_weight": 1.0, "nuisance_input_weight": 0.25},
    )
    assert gated.shape == xb.shape
    assert gated[0, 0].item() == xb[0, 0].item()
    assert gated[0, 1].item() == xb[0, 1].item() * 0.25


def test_learned_feature_gate_matches_fixed_gate_at_initialization() -> None:
    bundle = load_dataset({"seed": 3, "dataset": {"kind": "synthetic_linear", "n": 120}})
    xb = bundle.split("train")["x"][:4]
    fixed = _apply_causal_input_gate(
        bundle,
        xb,
        {"causal_input_weight": 1.0, "nuisance_input_weight": 0.25},
    )
    learned_gate = FeatureGate(
        bundle.causal_mask,
        causal_input_weight=1.0,
        nuisance_input_weight=0.25,
        learned=True,
    )
    learned = learned_gate(xb)
    assert learned.shape == fixed.shape
    assert learned.equal(fixed)


def test_grouped_feature_gate_shares_gate_within_group() -> None:
    bundle = load_dataset({"seed": 3, "dataset": {"kind": "synthetic_linear", "n": 120}})
    learned_gate = FeatureGate(
        bundle.causal_mask,
        causal_input_weight=1.0,
        nuisance_input_weight=0.25,
        learned=True,
        group_size=2,
    )
    with __import__("torch").no_grad():
        learned_gate.feature_logits[0] = 2.0
    gate = learned_gate.gate()
    base_gate = learned_gate.base_gate
    assert gate.shape[0] == bundle.input_dim
    assert (gate[0] / base_gate[0]).item() == (gate[1] / base_gate[1]).item()


def test_feature_gate_can_use_score_prior() -> None:
    import torch

    causal_mask = torch.tensor([1.0, 1.0], dtype=torch.float32)
    score_prior = torch.tensor([0.9, 0.1], dtype=torch.float32)
    gate = FeatureGate(
        causal_mask,
        learned=False,
        score_prior=score_prior,
        score_weight=0.5,
    )
    values = gate.gate()
    assert values[0].item() > values[1].item()


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


def test_gated_counterfactual_method_depends_on_mask_identity(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    rows = [
        "split,y,place,group,feature_0,feature_1,feature_2",
        "train,0,0,0,-1.0,-1.0,0.0",
        "train,0,0,0,-0.9,-0.9,0.1",
        "train,0,1,2,-1.1,-0.8,-0.1",
        "train,1,0,1,1.0,0.8,0.0",
        "train,1,1,3,0.9,1.0,0.1",
        "train,1,1,3,1.1,0.9,-0.1",
        "val,0,0,0,-1.0,-1.0,0.0",
        "val,0,1,2,-1.0,1.0,0.1",
        "val,1,0,1,1.0,-1.0,-0.1",
        "val,1,1,3,1.0,1.0,0.0",
        "test,0,0,0,-1.0,1.0,0.0",
        "test,0,1,2,-0.9,0.9,0.1",
        "test,1,0,1,1.0,-1.0,0.0",
        "test,1,1,3,0.9,-0.9,-0.1",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    training = {"device": "cpu", "epochs": 25, "batch_size": 4, "lr": 0.01}
    method = {
        "kind": "counterfactual_augmentation",
        "hidden_dim": 8,
        "consistency_weight": 0.2,
        "causal_input_weight": 1.0,
        "nuisance_input_weight": 0.0,
    }
    common_dataset = {
        "kind": "waterbirds_features",
        "path": str(csv_path),
    }

    correct_config = {
        "seed": 5,
        "dataset": {**common_dataset, "causal_feature_columns": ["feature_0"]},
        "method": method,
        "training": training,
    }
    wrong_config = {
        "seed": 5,
        "dataset": {**common_dataset, "causal_feature_columns": ["feature_1"]},
        "method": method,
        "training": training,
    }

    correct_bundle = load_dataset(correct_config)
    wrong_bundle = load_dataset(wrong_config)
    correct_model = fit_method(correct_bundle, correct_config)
    wrong_model = fit_method(wrong_bundle, wrong_config)
    correct_metrics = evaluate(correct_model, correct_bundle, correct_config)
    wrong_metrics = evaluate(wrong_model, wrong_bundle, wrong_config)

    assert correct_metrics["test/accuracy"] >= 0.99
    assert wrong_metrics["test/accuracy"] <= 0.51
    assert correct_metrics["test/accuracy"] - wrong_metrics["test/accuracy"] >= 0.45


def test_counterfactual_adversarial_runs_with_learned_input_gate(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    rows = [
        "split,y,place,group,feature_0,feature_1,feature_2",
        "train,0,0,0,-1.0,-1.0,0.0",
        "train,0,1,2,-1.1,0.9,0.2",
        "train,1,0,1,1.0,-0.9,-0.1",
        "train,1,1,3,1.1,1.0,0.0",
        "val,0,0,0,-1.0,-1.0,0.0",
        "val,1,1,3,1.0,1.0,0.0",
        "test,0,1,2,-1.0,1.0,0.1",
        "test,1,0,1,1.0,-1.0,-0.1",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    config = {
        "seed": 7,
        "dataset": {
            "kind": "waterbirds_features",
            "path": str(csv_path),
            "causal_feature_columns": ["feature_0"],
        },
        "method": {
            "kind": "counterfactual_adversarial",
            "hidden_dim": 8,
            "input_gate": "learned",
            "causal_input_weight": 1.0,
            "nuisance_input_weight": 0.5,
        },
        "training": {"device": "cpu", "epochs": 2, "batch_size": 4, "lr": 0.01},
    }
    bundle = load_dataset(config)
    model = fit_method(bundle, config)
    metrics = evaluate(model, bundle, config)
    assert 0.0 <= metrics["test/accuracy"] <= 1.0

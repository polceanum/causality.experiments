from pathlib import Path

import numpy as np
import pytest

from causality_experiments.data import load_dataset
from causality_experiments.methods import FeatureGate, _apply_causal_input_gate, _balanced_group_example_weights, _balanced_group_subsample_indices, _counterfactual_disagreement_weights, _estimate_counterfactual_instability, _fit_official_dfr_on_bundle, _instability_sample_weights, _select_replay_indices, _soft_nuisance_mask_from_scores, _update_instability_ema, fit_method
from causality_experiments.metrics import evaluate
from causality_experiments.sklearn_compat import LogisticRegression, StandardScaler


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


def test_feature_gate_can_learn_score_conditioned_offset() -> None:
    import torch

    causal_mask = torch.tensor([1.0, 1.0], dtype=torch.float32)
    score_prior = torch.tensor([0.9, 0.1], dtype=torch.float32)
    gate = FeatureGate(
        causal_mask,
        learned=False,
        score_prior=score_prior,
        score_conditioned=True,
    )
    with torch.no_grad():
        gate.score_alpha.fill_(2.0)
    values = gate.gate()
    assert values[0].item() > values[1].item()


def test_feature_gate_can_adapt_to_example_context() -> None:
    import torch

    causal_mask = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    gate = FeatureGate(
        causal_mask,
        learned=False,
        group_size=2,
        contextual=True,
    )
    with torch.no_grad():
        gate.context_alpha[0] = 2.0
    xb = torch.tensor(
        [
            [4.0, 4.0, 0.1, 0.1],
            [0.1, 0.1, 4.0, 4.0],
        ],
        dtype=torch.float32,
    )
    values = gate.gate(xb)
    assert values[0, 0].item() > values[0, 2].item()
    assert values[1, 0].item() < values[1, 2].item()


def test_feature_gate_can_adapt_to_representation_context() -> None:
    import torch

    causal_mask = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    gate = FeatureGate(
        causal_mask,
        learned=False,
        group_size=2,
        representation_conditioned=True,
        context_dim=3,
    )
    with torch.no_grad():
        gate.context_projection.weight.zero_()
        gate.context_projection.bias.zero_()
        gate.context_projection.weight[0, 0] = 2.0
        gate.context_projection.weight[1, 0] = -2.0
    context = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    values = gate.gate(context=context)
    assert values[0, 0].item() > values[0, 2].item()
    assert values[1, 0].item() < values[1, 2].item()


def test_counterfactual_disagreement_weights_emphasize_unstable_examples() -> None:
    import torch

    logits = torch.tensor([[4.0, 0.0], [0.5, 0.0]], dtype=torch.float32)
    cf_logits = torch.tensor([[0.0, 4.0], [0.45, 0.0]], dtype=torch.float32)
    weights = _counterfactual_disagreement_weights(logits, cf_logits, scale=1.0, floor=0.5)
    assert weights[0].item() > weights[1].item()
    assert weights[1].item() >= 0.5


def test_instability_ema_replay_prioritizes_persistently_unstable_examples() -> None:
    import torch

    ema = torch.zeros(4, dtype=torch.float32)
    ema = _update_instability_ema(
        ema,
        torch.tensor([0, 1, 2]),
        torch.tensor([0.8, 0.2, 0.5]),
        decay=0.5,
    )
    replay = _select_replay_indices(ema, torch.tensor([0, 1, 2]), fraction=1 / 3)
    assert replay.tolist() == [0]


def test_instability_sample_weights_upweight_top_fraction() -> None:
    import torch

    weights = _instability_sample_weights(
        torch.tensor([0.9, 0.1, 0.8, 0.2], dtype=torch.float32),
        top_fraction=0.5,
        upweight=5.0,
    )
    assert weights.tolist() == [5.0, 1.0, 5.0, 1.0]


def test_instability_score_mode_penalizes_high_variance_examples() -> None:
    import torch

    class StubModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, xb: torch.Tensor) -> torch.Tensor:
            base = torch.zeros((len(xb), 2), dtype=torch.float32)
            base[:, 0] = 4.0
            logits = [
                base.clone(),
                base.clone(),
                base.clone(),
            ]
            logits[1][0] = torch.tensor([0.0, 4.0], dtype=torch.float32)
            logits[1][1] = torch.tensor([2.0, 2.0], dtype=torch.float32)
            logits[2][1] = torch.tensor([0.0, 4.0], dtype=torch.float32)
            out = logits[self.calls]
            self.calls += 1
            return out

    bundle = load_dataset({"seed": 3, "dataset": {"kind": "synthetic_linear", "n": 120}})
    model = StubModel()
    mean_scores = _estimate_counterfactual_instability(model, bundle, passes=2, seed=0, score_mode="mean")
    model = StubModel()
    stable_scores = _estimate_counterfactual_instability(model, bundle, passes=2, seed=0, score_mode="mean_minus_std")
    assert (mean_scores[0] - stable_scores[0]).item() > (mean_scores[1] - stable_scores[1]).item()
    assert stable_scores[0].item() < stable_scores[1].item()


def test_instability_score_mode_can_upweight_hard_examples() -> None:
    import torch

    class StubModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, xb: torch.Tensor) -> torch.Tensor:
            base = torch.zeros((len(xb), 2), dtype=torch.float32)
            base[:, 0] = 4.0
            logits = [
                base.clone(),
                base.clone(),
                base.clone(),
            ]
            logits[0][1] = torch.tensor([0.3, 0.0], dtype=torch.float32)
            logits[1][0] = torch.tensor([0.0, 4.0], dtype=torch.float32)
            logits[1][1] = torch.tensor([0.0, 0.3], dtype=torch.float32)
            logits[2][0] = torch.tensor([0.0, 4.0], dtype=torch.float32)
            logits[2][1] = torch.tensor([0.0, 0.3], dtype=torch.float32)
            out = logits[self.calls]
            self.calls += 1
            return out

    bundle = load_dataset({"seed": 3, "dataset": {"kind": "synthetic_linear", "n": 120}})
    train = bundle.split("train")
    train["y"][:2] = torch.tensor([0, 1])
    model = StubModel()
    mean_scores = _estimate_counterfactual_instability(model, bundle, passes=2, seed=0, score_mode="mean")
    model = StubModel()
    weighted_scores = _estimate_counterfactual_instability(model, bundle, passes=2, seed=0, score_mode="loss_weighted_mean")
    assert (weighted_scores[0] / mean_scores[0]).item() < 1.0
    assert (weighted_scores[1] / mean_scores[1]).item() > (weighted_scores[0] / mean_scores[0]).item()
    assert weighted_scores[0].item() < weighted_scores[1].item()


def test_instability_score_mode_can_focus_on_counterfactual_loss_increase() -> None:
    import torch

    class StubModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, xb: torch.Tensor) -> torch.Tensor:
            factual = torch.zeros((len(xb), 2), dtype=torch.float32)
            factual[:, 0] = 4.0
            counterfactual = factual.clone()
            factual[1] = torch.tensor([1.2, 0.0], dtype=torch.float32)
            counterfactual[0] = torch.tensor([3.5, 0.0], dtype=torch.float32)
            counterfactual[1] = torch.tensor([0.0, 1.2], dtype=torch.float32)
            logits = [factual, counterfactual]
            out = logits[self.calls]
            self.calls += 1
            return out

    bundle = load_dataset({"seed": 3, "dataset": {"kind": "synthetic_linear", "n": 120}})
    train = bundle.split("train")
    train["y"][:2] = torch.tensor([0, 0])
    model = StubModel()
    loss_delta_scores = _estimate_counterfactual_instability(
        model,
        bundle,
        passes=1,
        seed=0,
        score_mode="counterfactual_loss_increase_mean",
    )
    assert loss_delta_scores[1].item() > loss_delta_scores[0].item()
    assert loss_delta_scores[1].item() > 0.0


def test_instability_score_mode_can_upweight_hard_groups() -> None:
    import torch

    class StubModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, xb: torch.Tensor) -> torch.Tensor:
            base = torch.zeros((len(xb), 2), dtype=torch.float32)
            base[:, 0] = 4.0
            logits = [
                base.clone(),
                base.clone(),
                base.clone(),
            ]
            logits[0][0] = torch.tensor([0.3, 0.0], dtype=torch.float32)
            logits[0][1] = torch.tensor([0.3, 0.0], dtype=torch.float32)
            logits[1][0] = torch.tensor([0.0, 4.0], dtype=torch.float32)
            logits[1][2] = torch.tensor([0.0, 4.0], dtype=torch.float32)
            out = logits[self.calls]
            self.calls += 1
            return out

    bundle = load_dataset({"seed": 3, "dataset": {"kind": "synthetic_linear", "n": 120}})
    train = bundle.split("train")
    train["y"][:] = 0
    train["group"][:] = 1
    train["y"][:2] = torch.tensor([1, 1])
    train["group"][:2] = 0
    model = StubModel()
    mean_scores = _estimate_counterfactual_instability(model, bundle, passes=2, seed=0, score_mode="mean")
    model = StubModel()
    group_scores = _estimate_counterfactual_instability(model, bundle, passes=2, seed=0, score_mode="group_loss_weighted_mean")
    assert round(mean_scores[0].item(), 6) == round(mean_scores[2].item(), 6)
    assert group_scores[0].item() > group_scores[2].item()


def test_instability_score_mode_can_focus_on_counterfactual_loss_increase_in_hard_groups() -> None:
    import torch

    class StubModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, xb: torch.Tensor) -> torch.Tensor:
            factual = torch.zeros((len(xb), 2), dtype=torch.float32)
            factual[:, 0] = 4.0
            counterfactual = factual.clone()
            factual[0] = torch.tensor([0.6, 0.0], dtype=torch.float32)
            factual[2] = torch.tensor([3.8, 0.0], dtype=torch.float32)
            counterfactual[0] = torch.tensor([0.0, 0.6], dtype=torch.float32)
            counterfactual[2] = torch.tensor([0.0, 0.6], dtype=torch.float32)
            outputs = [factual, counterfactual]
            out = outputs[self.calls]
            self.calls += 1
            return out

    bundle = load_dataset({"seed": 3, "dataset": {"kind": "synthetic_linear", "n": 120}})
    train = bundle.split("train")
    train["y"][:] = 0
    train["group"][:] = 1
    train["y"][[0, 2]] = torch.tensor([0, 0])
    train["group"][[0, 2]] = torch.tensor([0, 1])
    model = StubModel()
    scores = _estimate_counterfactual_instability(
        model,
        bundle,
        passes=1,
        seed=0,
        score_mode="group_loss_weighted_counterfactual_loss_increase_mean",
    )
    assert scores[0].item() > scores[2].item()
    assert scores[2].item() > 0.0


def test_instability_score_mode_rejects_unknown_value() -> None:
    class StubModel:
        def predict(self, xb):
            return xb[:, :2]

    bundle = load_dataset({"seed": 3, "dataset": {"kind": "synthetic_linear", "n": 120}})
    try:
        _estimate_counterfactual_instability(StubModel(), bundle, passes=1, seed=0, score_mode="mystery")
    except ValueError as exc:
        assert "Unknown counterfactual instability score mode" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown instability score mode")


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
    assert metrics["feature_importance/nuisance_to_causal"] == 0.0


def test_dfr_retrains_on_validation_split(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    rows = [
        "split,y,place,group,feature_0,feature_1",
        "train,0,0,0,1.0,0.0",
        "train,1,1,3,-1.0,0.0",
        "train,0,0,0,1.2,0.1",
        "train,1,1,3,-1.2,-0.1",
        "val,0,0,0,-1.0,0.0",
        "val,0,1,2,-1.2,0.1",
        "val,1,0,1,1.0,-0.1",
        "val,1,1,3,1.2,0.0",
        "test,0,1,2,-1.1,0.0",
        "test,1,0,1,1.1,0.0",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    config = {
        "seed": 5,
        "dataset": {"kind": "waterbirds_features", "path": str(csv_path)},
        "method": {"kind": "dfr", "dfr_epochs": 80, "dfr_lr": 0.05},
        "training": {"device": "cpu", "batch_size": 4},
    }
    bundle = load_dataset(config)
    model = fit_method(bundle, config)
    metrics = evaluate(model, bundle, config)
    assert metrics["test/accuracy"] >= 0.99


def test_dfr_group_weights_balance_examples() -> None:
    import torch

    weights = _balanced_group_example_weights(torch.tensor([0, 0, 0, 1]), power=1.0)
    flat_weights = _balanced_group_example_weights(torch.tensor([0, 0, 0, 1]), power=0.0)
    assert weights[3].item() > weights[0].item()
    assert abs(weights.mean().item() - 1.0) < 1e-6
    assert flat_weights.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_dfr_loss_weighted_group_mode_runs(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    rows = [
        "split,y,place,group,feature_0,feature_1",
        "train,0,0,0,1.0,0.0",
        "train,1,1,3,-1.0,0.0",
        "val,0,0,0,-1.0,0.0",
        "val,0,0,0,-1.1,0.1",
        "val,0,0,0,-1.2,0.0",
        "val,1,1,3,1.0,-0.1",
        "test,0,0,0,-1.0,0.0",
        "test,1,1,3,1.0,0.0",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    config = {
        "seed": 5,
        "dataset": {"kind": "waterbirds_features", "path": str(csv_path)},
        "method": {
            "kind": "dfr",
            "dfr_epochs": 40,
            "dfr_lr": 0.05,
            "dfr_group_weight_mode": "loss_weighted",
            "dfr_group_weight_power": 1.5,
        },
        "training": {"device": "cpu", "batch_size": 4},
    }
    bundle = load_dataset(config)
    model = fit_method(bundle, config)
    metrics = evaluate(model, bundle, config)
    assert metrics["test/accuracy"] >= 0.99


def test_dfr_lbfgs_optimizer_runs(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    rows = [
        "split,y,place,group,feature_0,feature_1",
        "train,0,0,0,1.0,0.0",
        "train,1,1,3,-1.0,0.0",
        "val,0,0,0,-1.0,0.0",
        "val,0,1,2,-1.2,0.1",
        "val,1,0,1,1.0,-0.1",
        "val,1,1,3,1.2,0.0",
        "test,0,1,2,-1.1,0.0",
        "test,1,0,1,1.1,0.0",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    config = {
        "seed": 5,
        "dataset": {"kind": "waterbirds_features", "path": str(csv_path)},
        "method": {
            "kind": "dfr",
            "dfr_optimizer": "lbfgs",
            "dfr_epochs": 20,
            "dfr_lr": 1.0,
            "dfr_weight_decay": 0.01,
        },
        "training": {"device": "cpu", "batch_size": 4},
    }
    bundle = load_dataset(config)
    model = fit_method(bundle, config)
    metrics = evaluate(model, bundle, config)
    assert metrics["test/accuracy"] >= 0.99


def test_dfr_can_train_on_combined_splits() -> None:
    config = {
        "seed": 3,
        "dataset": {"kind": "synthetic_linear", "n": 120},
        "method": {"kind": "dfr", "dfr_split": "train+val", "dfr_epochs": 2},
        "training": {"device": "cpu", "batch_size": 16, "lr": 0.01},
    }
    bundle = load_dataset(config)
    model = fit_method(bundle, config)
    predictions = model.predict(bundle.split("test")["x"])
    assert predictions.shape == (len(bundle.split("test")["x"]), bundle.output_dim)


def test_balanced_group_subsample_indices_equalize_counts_and_keep_smallest_group() -> None:
    groups = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    idx = _balanced_group_subsample_indices(groups, np.random.default_rng(7))
    selected_groups = groups[idx]
    counts = {group: int((selected_groups == group).sum()) for group in np.unique(selected_groups)}
    assert counts == {0: 2, 1: 2, 2: 2}
    assert set(np.flatnonzero(groups == 0)).issubset(set(idx.tolist()))


def test_official_dfr_val_tr_uses_only_validation_reweighting_and_subset_scaler(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    rows = [
        "split,y,place,group,feature_0,feature_1",
        "train,0,0,0,1000.0,0.0",
        "train,1,1,3,1002.0,0.0",
        "train,0,1,2,998.0,0.0",
        "train,1,0,1,1004.0,0.0",
        "val,0,0,0,-4.0,0.0",
        "val,0,0,0,-3.8,0.1",
        "val,1,0,1,3.8,-0.1",
        "val,1,0,1,4.0,0.0",
        "val,0,1,2,-4.2,0.0",
        "val,0,1,2,-4.1,0.1",
        "val,1,1,3,4.1,-0.1",
        "val,1,1,3,4.2,0.0",
        "test,0,0,0,-4.0,0.0",
        "test,1,1,3,4.0,0.0",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    config = {
        "seed": 11,
        "dataset": {"kind": "waterbirds_features", "path": str(csv_path)},
        "method": {
            "kind": "official_dfr_val_tr",
            "official_dfr_c_grid": [1.0],
            "official_dfr_num_retrains": 1,
            "official_dfr_balance_val": True,
            "official_dfr_add_train": False,
        },
    }
    bundle = load_dataset(config)
    model = fit_method(bundle, config)
    details = getattr(model, "details", {})
    assert details["official_dfr_add_train"] is False
    assert details["official_dfr_retrains"][0]["train_indices"] == []
    tune_val_indices = details["official_dfr_tune_val_indices"]
    val_x = bundle.split("val")["x"][tune_val_indices].numpy()
    expected_mean = val_x.mean(axis=0)
    np.testing.assert_allclose(details["official_dfr_tune_scaler_mean"], expected_mean, atol=1e-6)
    assert abs(details["official_dfr_tune_scaler_mean"][0]) < 10.0


def test_official_dfr_val_tr_balances_groups_and_is_deterministic(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    rows = [
        "split,y,place,group,feature_0,feature_1",
        "train,0,0,0,-1.0,0.0",
        "train,1,1,3,1.0,0.0",
        "val,0,0,0,-2.0,0.0",
        "val,0,0,0,-1.9,0.0",
        "val,0,0,0,-1.8,0.0",
        "val,1,0,1,1.8,0.0",
        "val,1,0,1,1.9,0.0",
        "val,0,1,2,-2.2,0.0",
        "val,0,1,2,-2.1,0.0",
        "val,1,1,3,2.0,0.0",
        "test,0,0,0,-2.0,0.0",
        "test,1,1,3,2.0,0.0",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    config = {
        "seed": 3,
        "dataset": {"kind": "waterbirds_features", "path": str(csv_path)},
        "method": {
            "kind": "official_dfr_val_tr",
            "official_dfr_c_grid": [1.0],
            "official_dfr_num_retrains": 3,
            "official_dfr_balance_val": True,
            "official_dfr_add_train": False,
        },
    }
    bundle = load_dataset(config)
    model_a = fit_method(bundle, config)
    model_b = fit_method(bundle, config)
    np.testing.assert_allclose(model_a.weight.numpy(), model_b.weight.numpy(), atol=1e-7)
    np.testing.assert_allclose(model_a.bias.numpy(), model_b.bias.numpy(), atol=1e-7)
    for retrain in model_a.details["official_dfr_retrains"]:
        counts = retrain["val_group_counts"]
        assert len(set(counts.values())) == 1


def test_official_dfr_val_tr_matches_direct_sklearn_retraining(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    rows = [
        "split,y,place,group,feature_0,feature_1",
        "train,0,0,0,-1.0,0.0",
        "train,1,1,3,1.0,0.0",
        "val,0,0,0,-3.0,0.0",
        "val,0,0,0,-2.8,0.0",
        "val,1,0,1,2.8,0.0",
        "val,1,0,1,3.0,0.0",
        "val,0,1,2,-3.2,0.0",
        "val,0,1,2,-3.1,0.0",
        "val,1,1,3,3.1,0.0",
        "val,1,1,3,3.2,0.0",
        "test,0,0,0,-3.0,0.0",
        "test,1,1,3,3.0,0.0",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    config = {
        "seed": 5,
        "dataset": {"kind": "waterbirds_features", "path": str(csv_path)},
        "method": {
            "kind": "official_dfr_val_tr",
            "official_dfr_c_grid": [0.3],
            "official_dfr_num_retrains": 1,
            "official_dfr_balance_val": True,
            "official_dfr_add_train": False,
        },
    }
    bundle = load_dataset(config)
    model = fit_method(bundle, config)
    details = model.details["official_dfr_retrains"][0]
    val_x = bundle.split("val")["x"].numpy()
    val_y = bundle.split("val")["y"].numpy()
    fit_x = val_x[details["val_indices"]]
    fit_y = val_y[details["val_indices"]]
    scaler = StandardScaler()
    fit_x_scaled = scaler.fit_transform(fit_x)
    logreg = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=0.3,
        max_iter=1000,
        random_state=5,
    )
    logreg.fit(fit_x_scaled, fit_y)
    test_x = bundle.split("test")["x"].numpy()
    direct_pred = logreg.predict(scaler.transform(test_x))
    model_pred = model.predict(bundle.split("test")["x"]).argmax(dim=1).numpy()
    assert direct_pred.tolist() == model_pred.tolist()


def test_official_causal_shrink_identity_matches_official_dfr(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    rows = [
        "split,y,place,group,feature_0,feature_1",
        "train,0,0,0,-1.0,-1.0",
        "train,1,1,3,1.0,1.0",
        "val,0,0,0,-3.0,-2.0",
        "val,0,0,0,-2.8,-1.8",
        "val,1,0,1,2.8,1.8",
        "val,1,0,1,3.0,2.0",
        "val,0,1,2,-3.2,-2.2",
        "val,0,1,2,-3.1,-2.1",
        "val,1,1,3,3.1,2.1",
        "val,1,1,3,3.2,2.2",
        "test,0,0,0,-3.0,-2.0",
        "test,1,1,3,3.0,2.0",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    base_config = {
        "seed": 5,
        "dataset": {"kind": "waterbirds_features", "path": str(csv_path), "causal_feature_columns": ["feature_0"]},
        "method": {
            "kind": "official_dfr_val_tr",
            "official_dfr_c_grid": [0.3],
            "official_dfr_num_retrains": 1,
            "official_dfr_balance_val": True,
            "official_dfr_add_train": False,
        },
    }
    shrink_config = {**base_config, "method": {**base_config["method"], "kind": "official_causal_shrink_dfr_val_tr", "official_causal_shrink_grid": [1.0]}}
    bundle = load_dataset(base_config)
    official = fit_method(bundle, base_config)
    shrink = fit_method(bundle, shrink_config)
    np.testing.assert_allclose(shrink.weight.numpy(), official.weight.numpy(), atol=1e-7)
    np.testing.assert_allclose(shrink.bias.numpy(), official.bias.numpy(), atol=1e-7)
    assert shrink.details["official_dfr_best_feature_scale"] == 1.0


def test_official_causal_shrink_can_change_nuisance_coefficients(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    rows = [
        "split,y,place,group,feature_0,feature_1",
        "train,0,0,0,-1.0,-1.0",
        "train,1,1,3,1.0,1.0",
        "val,0,0,0,-0.2,-3.5",
        "val,0,0,0,0.2,-1.4",
        "val,1,0,1,0.1,1.6",
        "val,1,0,1,-0.1,3.4",
        "val,0,1,2,-0.1,-4.0",
        "val,0,1,2,0.1,-2.1",
        "val,1,1,3,0.2,2.4",
        "val,1,1,3,-0.2,4.1",
        "test,0,0,0,-0.1,-2.5",
        "test,1,1,3,0.1,2.5",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    config = {
        "seed": 5,
        "dataset": {"kind": "waterbirds_features", "path": str(csv_path), "causal_feature_columns": ["feature_0"]},
        "method": {
            "kind": "official_dfr_val_tr",
            "official_dfr_c_grid": [10.0],
            "official_dfr_num_retrains": 1,
            "official_dfr_balance_val": True,
            "official_dfr_add_train": False,
        },
    }
    bundle = load_dataset(config)
    official = fit_method(bundle, config)
    shrunk = _fit_official_dfr_on_bundle(
        bundle,
        config,
        feature_scale_grid=[(0.25, np.array([1.0, 0.25], dtype=np.float64))],
    )
    assert not np.allclose(shrunk.weight.numpy(), official.weight.numpy(), atol=1e-7)
    assert abs(float(shrunk.weight.numpy()[0, 1])) < abs(float(official.weight.numpy()[0, 1]))


def test_official_dfr_accepts_example_weight_key(tmp_path: Path) -> None:
    import torch

    csv_path = tmp_path / "features.csv"
    rows = [
        "split,y,place,group,feature_0,feature_1",
        "train,0,0,0,-1.0,0.0",
        "train,1,1,3,1.0,0.0",
        "val,0,0,0,-1.0,0.0",
        "val,0,0,0,-0.8,0.2",
        "val,1,0,1,0.8,0.1",
        "val,1,0,1,1.0,0.3",
        "val,0,1,2,-0.9,-0.1",
        "val,0,1,2,-0.7,-0.2",
        "val,1,1,3,0.7,-0.3",
        "val,1,1,3,0.9,-0.4",
        "test,0,0,0,-1.0,0.0",
        "test,1,1,3,1.0,0.0",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    config = {
        "seed": 5,
        "dataset": {"kind": "waterbirds_features", "path": str(csv_path)},
        "method": {
            "kind": "official_dfr_val_tr",
            "official_dfr_c_grid": [1.0],
            "official_dfr_num_retrains": 1,
            "official_dfr_balance_val": True,
            "official_dfr_example_weight_key": "official_dfr_weight",
        },
    }
    bundle = load_dataset(config)
    for split in bundle.splits.values():
        split["official_dfr_weight"] = torch.ones(len(split["y"]), dtype=torch.float32)

    weighted = fit_method(bundle, config)

    assert weighted.details["official_dfr_example_weight_key"] == "official_dfr_weight"


def test_dfr_rejects_unknown_group_weight_mode() -> None:
    config = {
        "seed": 3,
        "dataset": {"kind": "synthetic_linear", "n": 120},
        "method": {"kind": "dfr", "dfr_group_weight_mode": "mystery"},
        "training": {"device": "cpu", "epochs": 1},
    }
    bundle = load_dataset(config)
    try:
        fit_method(bundle, config)
    except ValueError as exc:
        assert "dfr_group_weight_mode" in str(exc)
    else:
        raise AssertionError("expected invalid DFR group weight mode to fail")


def test_dfr_counterfactual_consistency_requires_causal_mask(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    rows = [
        "split,y,place,group,feature_0,feature_1",
        "train,0,0,0,-1.0,0.0",
        "train,1,1,3,1.0,0.0",
        "val,0,0,0,-1.0,0.0",
        "val,1,1,3,1.0,0.0",
        "test,0,0,0,-1.0,0.0",
        "test,1,1,3,1.0,0.0",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    config = {
        "seed": 5,
        "dataset": {"kind": "waterbirds_features", "path": str(csv_path)},
        "method": {"kind": "dfr", "dfr_counterfactual_consistency_weight": 1.0},
        "training": {"device": "cpu"},
    }
    bundle = load_dataset(config)
    try:
        fit_method(bundle, config)
    except ValueError as exc:
        assert "DFR counterfactual consistency requires dataset.causal_mask" in str(exc)
    else:
        raise AssertionError("expected DFR counterfactual consistency to require a causal mask")


def test_causal_dfr_requires_causal_mask(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    rows = [
        "split,y,place,group,feature_0,feature_1",
        "train,0,0,0,-1.0,0.0",
        "train,1,1,3,1.0,0.0",
        "val,0,0,0,-1.0,0.0",
        "val,1,1,3,1.0,0.0",
        "test,0,0,0,-1.0,0.0",
        "test,1,1,3,1.0,0.0",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    config = {
        "seed": 5,
        "dataset": {"kind": "waterbirds_features", "path": str(csv_path)},
        "method": {"kind": "causal_dfr"},
        "training": {"device": "cpu"},
    }
    bundle = load_dataset(config)
    try:
        fit_method(bundle, config)
    except ValueError as exc:
        assert "Causal DFR requires dataset.causal_mask" in str(exc)
    else:
        raise AssertionError("expected causal_dfr to require a causal mask")


def test_causal_dfr_suppresses_nuisance_dimensions(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    rows = [
        "split,y,place,group,feature_0,feature_1",
        "train,0,0,0,-1.0,-1.0",
        "train,1,1,3,1.0,1.0",
        "val,0,0,0,-1.0,-1.0",
        "val,0,1,2,-1.1,-1.1",
        "val,1,0,1,1.0,1.0",
        "val,1,1,3,1.1,1.1",
        "test,0,0,0,-1.0,-1.0",
        "test,1,1,3,1.0,1.0",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    config = {
        "seed": 5,
        "dataset": {
            "kind": "waterbirds_features",
            "path": str(csv_path),
            "causal_feature_columns": ["feature_0"],
        },
        "method": {
            "kind": "causal_dfr",
            "dfr_epochs": 120,
            "dfr_lr": 0.05,
            "causal_dfr_nuisance_weight": 20.0,
        },
        "training": {"device": "cpu", "batch_size": 4},
    }
    bundle = load_dataset(config)
    model = fit_method(bundle, config)
    metrics = evaluate(model, bundle, config)
    importance = model.feature_importance()
    assert importance is not None
    assert importance[1].item() < importance[0].item() * 0.2
    assert metrics["feature_importance/nuisance_to_causal"] < 0.2


def test_causal_dfr_can_average_multiple_retrains(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    rows = [
        "split,y,place,group,feature_0,feature_1",
        "train,0,0,0,-1.0,-0.8",
        "train,1,1,3,1.0,0.8",
        "val,0,0,0,-1.0,-1.0",
        "val,0,1,2,-1.1,1.0",
        "val,1,0,1,1.0,-1.0",
        "val,1,1,3,1.1,1.0",
        "test,0,0,0,-1.0,-0.5",
        "test,1,1,3,1.0,0.5",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    config = {
        "seed": 5,
        "dataset": {
            "kind": "waterbirds_features",
            "path": str(csv_path),
            "causal_feature_columns": ["feature_0"],
        },
        "method": {
            "kind": "causal_dfr",
            "dfr_epochs": 3,
            "dfr_lr": 0.05,
            "dfr_num_retrains": 2,
            "causal_dfr_nuisance_weight": 1.0,
        },
        "training": {"device": "cpu", "batch_size": 4},
    }
    bundle = load_dataset(config)
    model = fit_method(bundle, config)
    predictions = model.predict(bundle.split("test")["x"])
    importance = model.feature_importance()
    assert predictions.shape == (2, 2)
    assert importance is not None
    assert importance.shape[0] == bundle.input_dim


def test_representation_dfr_trains_head_on_learned_representations() -> None:
    config = {
        "seed": 3,
        "dataset": {"kind": "synthetic_linear", "n": 180},
        "method": {
            "kind": "representation_dfr",
            "hidden_dim": 8,
            "representation_epochs": 20,
            "dfr_epochs": 40,
            "dfr_lr": 0.05,
            "dfr_weight_decay": 0.001,
        },
        "training": {"device": "cpu", "epochs": 20, "batch_size": 32, "lr": 0.01},
    }
    bundle = load_dataset(config)
    model = fit_method(bundle, config)
    metrics = evaluate(model, bundle, config)
    importance = model.feature_importance()
    assert 0.0 <= metrics["test/accuracy"] <= 1.0
    assert importance is not None
    assert importance.shape[0] == bundle.input_dim


def test_official_representation_dfr_trains_official_head_on_learned_representations() -> None:
    config = {
        "seed": 7,
        "dataset": {"kind": "synthetic_linear", "n": 180},
        "method": {
            "kind": "official_representation_dfr",
            "representation_method": "adversarial_probe",
            "hidden_dim": 8,
            "representation_epochs": 20,
            "representation_lr": 0.01,
            "adv_weight": 0.1,
            "official_dfr_num_retrains": 3,
            "official_dfr_balance_val": True,
            "official_dfr_add_train": False,
        },
        "training": {"device": "cpu", "batch_size": 32, "epochs": 20, "lr": 0.01},
    }
    bundle = load_dataset(config)
    model = fit_method(bundle, config)
    metrics = evaluate(model, bundle, config)
    importance = model.feature_importance()
    assert 0.0 <= metrics["test/accuracy"] <= 1.0
    assert importance is not None
    assert importance.shape[0] == bundle.input_dim


def test_soft_score_causal_dfr_prior_uses_metadata_scores() -> None:
    bundle = load_dataset({"seed": 3, "dataset": {"kind": "synthetic_linear", "n": 120}})
    soft_nuisance = _soft_nuisance_mask_from_scores(
        bundle.__class__(
            name=bundle.name,
            task=bundle.task,
            splits=bundle.splits,
            input_dim=bundle.input_dim,
            output_dim=bundle.output_dim,
            causal_mask=bundle.causal_mask,
            metadata={**(bundle.metadata or {}), "causal_feature_scores": [0.9, 0.1]},
        )
    )
    assert soft_nuisance[0].item() < soft_nuisance[1].item()


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
        "dfr",
        "causal_dfr",
        "representation_dfr",
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


def test_adversarial_probe_respects_fixed_causal_input_gate(tmp_path: Path) -> None:
    csv_path = tmp_path / "adv_features.csv"
    rows = [
        "split,y,place,group,feature_0,feature_1",
        "train,0,0,0,-1.0,-1.0",
        "train,0,1,2,-1.1,1.0",
        "train,1,0,1,1.0,-1.0",
        "train,1,1,3,1.1,1.0",
        "val,0,0,0,-1.0,-1.0",
        "val,1,1,3,1.0,1.0",
        "test,0,1,2,-1.0,1.0",
        "test,1,0,1,1.0,-1.0",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    training = {"device": "cpu", "epochs": 25, "batch_size": 4, "lr": 0.01}
    method = {
        "kind": "adversarial_probe",
        "hidden_dim": 8,
        "adv_weight": 0.1,
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


def test_adversarial_probe_score_guided_gate_uses_feature_scores(tmp_path: Path) -> None:
    csv_path = tmp_path / "score_guided_features.csv"
    rows = [
        "split,y,place,group,feature_0,feature_1",
        "train,0,0,0,-1.0,-1.0",
        "train,0,1,2,-1.1,1.0",
        "train,1,0,1,1.0,-1.0",
        "train,1,1,3,1.1,1.0",
        "val,0,0,0,-1.0,-1.0",
        "val,1,1,3,1.0,1.0",
        "test,0,1,2,-1.0,1.0",
        "test,1,0,1,1.0,-1.0",
    ]
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    training = {"device": "cpu", "epochs": 25, "batch_size": 4, "lr": 0.01}
    method = {
        "kind": "adversarial_probe",
        "hidden_dim": 8,
        "adv_weight": 0.1,
        "input_gate_score_only": True,
        "causal_input_weight": 1.0,
        "nuisance_input_weight": 0.0,
    }
    common_dataset = {"kind": "waterbirds_features", "path": str(csv_path)}
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


def test_adversarial_probe_nuisance_regularizer_requires_feature_scores() -> None:
    config = {
        "seed": 3,
        "dataset": {"kind": "synthetic_linear", "n": 120},
        "method": {
            "kind": "adversarial_probe",
            "hidden_dim": 8,
            "representation_nuisance_penalty_weight": 0.1,
        },
        "training": {"device": "cpu", "epochs": 2, "batch_size": 16, "lr": 0.01},
    }
    bundle = load_dataset(config)
    with pytest.raises(ValueError, match="causal_feature_scores"):
        fit_method(bundle, config)

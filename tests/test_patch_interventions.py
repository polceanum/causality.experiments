import torch
import pytest

from causality_experiments.patch_interventions import (
    PatchFlipProbe,
    counterfactual_probe_loss,
    flipped_binary_targets,
    intervention_discovery_score,
    patch_selector_scores,
    patch_tokens_from_hidden,
    replace_hidden_patch_tokens,
    replace_hidden_patch_tokens_soft,
    replace_patch_tokens,
    soft_patch_mask,
    summarize_counterfactual_effects,
    topk_patch_mask,
)


def test_patch_selector_scores_use_cls_similarity() -> None:
    hidden = torch.tensor(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    scores = patch_selector_scores(hidden, "cls_similarity")
    mask = topk_patch_mask(scores, top_k=1)

    assert torch.equal(patch_tokens_from_hidden(hidden), hidden[:, 1:])
    assert scores.argmax(dim=1).tolist() == [0]
    assert mask.tolist() == [[True, False, False]]


def test_replace_patch_tokens_supports_donor_and_prototype() -> None:
    patches = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    donor = torch.tensor([[[9.0, 9.0], [8.0, 8.0], [7.0, 7.0]]])
    mask = torch.tensor([[False, True, False]])

    donor_edited = replace_patch_tokens(patches, mask, replacement="donor", donor_patches=donor)
    prototype_edited = replace_patch_tokens(patches, mask, replacement="prototype", prototype=torch.tensor([5.0, 6.0]))

    assert donor_edited.tolist() == [[[1.0, 1.0], [8.0, 8.0], [3.0, 3.0]]]
    assert prototype_edited.tolist() == [[[1.0, 1.0], [5.0, 6.0], [3.0, 3.0]]]


def test_replace_hidden_patch_tokens_preserves_cls_token() -> None:
    hidden = torch.tensor([[[100.0, 100.0], [1.0, 1.0], [2.0, 2.0]]])
    mask = torch.tensor([[True, False]])

    edited = replace_hidden_patch_tokens(hidden, mask, replacement="zero")

    assert edited[:, :1].tolist() == [[[100.0, 100.0]]]
    assert edited[:, 1:].tolist() == [[[0.0, 0.0], [2.0, 2.0]]]


def test_summarize_counterfactual_effects_reports_flips_and_group_effects() -> None:
    baseline = torch.tensor([[0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    edited = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
    labels = torch.tensor([1, 0, 1])
    groups = torch.tensor([0, 0, 1])

    summary = summarize_counterfactual_effects(baseline, edited, labels, groups=groups)

    assert summary["prediction_flip_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["correct_to_wrong_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["mean_target_logit_delta"] == pytest.approx(-2.0 / 3.0)
    assert summary["group_mean_target_logit_delta"] == {"0": -1.5, "1": 1.0}


def test_intervention_discovery_score_rewards_label_specific_effects() -> None:
    causal = intervention_discovery_score(label_effect=2.0, background_effect=0.1, random_control_effect=0.1)
    nuisance = intervention_discovery_score(label_effect=0.2, background_effect=2.0, random_control_effect=0.1)

    assert causal > nuisance
    assert causal > 0.5


def test_patch_flip_probe_scores_each_patch() -> None:
    probe = PatchFlipProbe(token_dim=2, hidden_dim=4)
    hidden = torch.randn(3, 5, 2)

    scores = probe(hidden)

    assert scores.shape == (3, 4)


def test_patch_flip_probe_can_initialize_to_mask_budget() -> None:
    probe = PatchFlipProbe(token_dim=2, initial_mask_probability=0.1)
    hidden = torch.randn(3, 5, 2)

    mask = soft_patch_mask(probe(hidden))

    assert float(mask.mean().item()) == pytest.approx(0.1)


def test_soft_patch_replacement_interpolates_patch_values() -> None:
    hidden = torch.tensor([[[10.0, 10.0], [1.0, 1.0], [3.0, 3.0]]])
    weights = torch.tensor([[0.0, 0.5]])

    edited = replace_hidden_patch_tokens_soft(hidden, weights, replacement="mean")

    assert edited[:, :1].tolist() == [[[10.0, 10.0]]]
    assert edited[:, 1:].tolist() == [[[1.0, 1.0], [2.5, 2.5]]]


def test_counterfactual_probe_loss_uses_flipped_predictions_and_mask_penalty() -> None:
    baseline = torch.tensor([[0.0, 2.0], [3.0, 0.0]])
    edited = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    mask_logits = torch.tensor([[2.0, -2.0], [0.0, 0.0]])
    mask = soft_patch_mask(mask_logits, temperature=1.0)

    targets = flipped_binary_targets(baseline, mode="prediction")
    loss, parts = counterfactual_probe_loss(edited, targets, mask, sparsity_weight=0.5, budget=0.25, budget_weight=1.0)

    assert targets.tolist() == [0, 1]
    assert loss.item() > parts["flip_loss"]
    assert parts["sparsity_loss"] == pytest.approx(float(mask.mean().item()))

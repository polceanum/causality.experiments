from pathlib import Path

from causality_experiments.counterfactual_clue_tests import clue_rows_from_test_results, execute_clue_tests
from causality_experiments.data import load_dataset
from causality_experiments.latent_clue_packets import build_latent_clue_packets, packets_to_jsonl
from causality_experiments.llm_clue_bridge import build_bridge_training_rows, hypothesis_label_from_packet
from causality_experiments.llm_clue_planner import (
    CluePlannerBackend,
    MockCluePlannerBackend,
    parse_clue_plan,
    plan_from_backend,
    render_planner_prompt,
)
from causality_experiments.rl_clue_policy import (
    assert_no_benchmark_final_training,
    build_clue_reward_rows,
    clue_reward_row,
    score_policy_packets,
    train_offline_clue_policy,
)
from scripts.run_llm_counterfactual_clue_probe import run_llm_counterfactual_clue_probe
from scripts.train_llm_clue_policy import run_clue_policy_training
from scripts.train_llm_clue_bridge_ranker import run_bridge_ranker


def _write_waterbirds_features(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "split,y,place,bird_shape_0,background_0,mixed_0",
                "train,0,0,0.0,0.0,0.0",
                "train,0,1,0.0,1.0,0.5",
                "train,1,0,1.0,0.0,0.5",
                "train,1,1,1.0,1.0,1.0",
                "val,0,0,0.0,0.0,0.0",
                "val,1,1,1.0,1.0,1.0",
                "test,0,1,0.0,1.0,0.5",
                "test,1,0,1.0,0.0,0.5",
            ]
        ),
        encoding="utf-8",
    )


def _load_bundle(path: Path):
    return load_dataset(
        {
            "dataset": {
                "kind": "waterbirds_features",
                "path": str(path),
                "causal_feature_columns": ["bird_shape_0"],
            }
        }
    )


def test_latent_clue_packets_capture_feature_and_probe_evidence(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    _write_waterbirds_features(csv_path)
    bundle = _load_bundle(csv_path)

    packets = build_latent_clue_packets(
        bundle,
        top_k=2,
        probe_summary={"probe/selectivity": 0.25},
    )

    by_name = {str(packet["feature_name"]): packet for packet in packets}
    bird = by_name["bird_shape_0"]
    background = by_name["background_0"]
    assert bird["feature_group"] == "label"
    assert background["feature_group"] == "background"
    assert bird["probe_summary"] == {"probe/selectivity": 0.25}
    assert bird["packet_hash"]
    assert packets_to_jsonl([bird]).startswith('{"')


def test_mock_planner_selects_trainable_actions_from_latent_packets(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    _write_waterbirds_features(csv_path)
    packets = build_latent_clue_packets(_load_bundle(csv_path), top_k=2)

    plan = plan_from_backend(packets, MockCluePlannerBackend(), max_packets=3)

    by_feature = {hypothesis.feature_name: hypothesis for hypothesis in plan.hypotheses}
    tests_by_feature = {test.feature_name: test for test in plan.tests}
    assert by_feature["bird_shape_0"].hypothesis_type == "causal"
    assert by_feature["background_0"].hypothesis_type == "nuisance"
    assert tests_by_feature["bird_shape_0"].action == "feature_mean_ablation"
    assert tests_by_feature["background_0"].action == "donor_swap_same_label_diff_env"
    assert plan.backend == "mock"
    assert not plan.fallback


def test_planner_repairs_json_wrappers_and_falls_back_on_invalid_backend(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    _write_waterbirds_features(csv_path)
    packets = build_latent_clue_packets(_load_bundle(csv_path), top_k=2, max_packets=1)
    prompt = render_planner_prompt(packets)
    wrapped = "Here is the plan: " + MockCluePlannerBackend().complete(prompt)

    repaired = parse_clue_plan(wrapped, backend="wrapped")

    assert repaired.repaired
    assert repaired.hypotheses

    class BrokenBackend:
        name = "broken"

        def complete(self, prompt: str) -> str:
            return "not json"

    fallback = plan_from_backend(packets, BrokenBackend(), max_packets=1)
    assert fallback.fallback
    assert fallback.backend == "mock"
    assert fallback.tests


def test_bridge_training_rows_create_targets_from_packets_and_results(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    _write_waterbirds_features(csv_path)
    packets = build_latent_clue_packets(_load_bundle(csv_path), top_k=2)
    plan = plan_from_backend(packets, MockCluePlannerBackend(), max_packets=3)
    bird_test = next(test for test in plan.tests if test.feature_name == "bird_shape_0")
    results = [
        {
            "candidate_id": bird_test.candidate_id,
            "test_effect_label_delta": 0.5,
            "test_effect_env_delta": 0.1,
            "test_effect_selectivity": 0.2,
            "test_random_control_delta": 0.05,
            "test_passed_control": True,
            "test_cost": 1.0,
        }
    ]

    rows = build_bridge_training_rows(packets, plan.hypotheses, plan.tests, results)

    by_feature = {str(row["feature_name"]): row for row in rows}
    assert hypothesis_label_from_packet(next(packet for packet in packets if packet["feature_name"] == "bird_shape_0")) == "causal"
    assert by_feature["bird_shape_0"]["target_hypothesis_label"] == "causal"
    assert by_feature["bird_shape_0"]["hypothesis_correct"] is True
    assert by_feature["bird_shape_0"]["test_value"] > 0.0
    assert by_feature["bird_shape_0"]["score_delta"] > 0.0
    assert by_feature["background_0"]["target_hypothesis_label"] == "nuisance"


def test_clue_reward_rows_expose_trainable_offline_rewards(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    _write_waterbirds_features(csv_path)
    packets = build_latent_clue_packets(_load_bundle(csv_path), top_k=2)
    good_packet = next(packet for packet in packets if packet["feature_name"] == "bird_shape_0")
    trace = {
        "candidate_id": good_packet["candidate_id"],
        "feature_name": good_packet["feature_name"],
        "action": "feature_mean_ablation",
        "test_value": 0.4,
        "score_delta": 0.2,
        "passed_control": True,
        "hypothesis_correct": True,
    }

    rows = build_clue_reward_rows(
        packets=packets,
        traces=[trace],
        feature_clues={"bird_shape_0": {"causal_target": "1.0"}},
        dataset="fixture",
    )

    assert rows[0]["reward_schema_version"] == "rl_clue_reward/v1"
    assert rows[0]["trainable_reward"] is True
    assert rows[0]["total_reward"] > rows[0]["test_reward"]


def test_clue_reward_rows_block_benchmark_final_training() -> None:
    row = clue_reward_row(
        packet={"feature_name": "feature_0", "label_corr": 1.0, "env_corr": 0.0},
        trace={"action": "feature_mean_ablation"},
        reward_scope="benchmark_final",
    )

    try:
        assert_no_benchmark_final_training([row])
    except ValueError as exc:
        assert "Benchmark-final" in str(exc)
    else:
        raise AssertionError("Expected benchmark-final rewards to be blocked from policy training.")

    try:
        train_offline_clue_policy([row])
    except ValueError as exc:
        assert "Benchmark-final" in str(exc)
    else:
        raise AssertionError("Expected policy training to reject benchmark-final rewards.")


def test_offline_clue_policy_scores_packets_by_rewarded_actions(tmp_path: Path) -> None:
    good = {
        "feature_index": 0,
        "feature_name": "feature_good",
        "label_corr": 0.9,
        "env_corr": 0.1,
        "corr_margin": 0.8,
        "abs_corr_margin": 0.8,
        "uncertainty": 0.1,
        "top_group_entropy": 0.2,
        "label_env_disentanglement": 0.8,
    }
    bad = {
        "feature_index": 1,
        "feature_name": "feature_bad",
        "label_corr": 0.1,
        "env_corr": 0.8,
        "corr_margin": -0.7,
        "abs_corr_margin": 0.7,
        "uncertainty": 0.9,
        "top_group_entropy": 0.8,
        "label_env_disentanglement": 0.1,
    }
    rows = [
        clue_reward_row(
            packet=good,
            trace={"action": "feature_mean_ablation", "test_value": 1.0, "score_delta": 0.5, "passed_control": True},
        ),
        clue_reward_row(
            packet=bad,
            trace={"action": "donor_swap_same_label_diff_env", "test_value": 0.0, "score_delta": 0.0},
        ),
    ]

    policy = train_offline_clue_policy(rows, alpha=0.1)
    scores = score_policy_packets([good, bad], policy)
    by_feature = {row["feature_name"]: float(row["score"]) for row in scores}

    assert by_feature["feature_good"] > by_feature["feature_bad"]


def test_counterfactual_clue_tests_execute_model_effects_and_controls(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    _write_waterbirds_features(csv_path)
    bundle = _load_bundle(csv_path)
    packets = build_latent_clue_packets(bundle, top_k=2)
    plan = plan_from_backend(packets, MockCluePlannerBackend(), max_packets=3)

    from causality_experiments.methods import fit_method

    model = fit_method(bundle, {"method": {"kind": "oracle"}})
    results = execute_clue_tests(bundle, plan.tests, packets=packets, model=model, split_name="test")

    by_feature = {str(row["feature_name"]): row for row in results}
    bird = by_feature["bird_shape_0"]
    assert bird["test_effect_label_delta"] > bird["test_random_control_delta"]
    assert bird["test_passed_control"] is True
    clue_rows = clue_rows_from_test_results(results)
    by_clue = {str(row["feature_name"]): row for row in clue_rows}
    assert by_clue["bird_shape_0"]["llm_untested"] == "0"
    assert by_clue["bird_shape_0"]["test_passed_control"] == "1"


def test_llm_counterfactual_clue_probe_runner_writes_replayable_artifacts(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    _write_waterbirds_features(csv_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: llm_probe_test",
                "dataset:",
                "  kind: waterbirds_features",
                f"  path: {csv_path}",
                "  causal_feature_columns:",
                "    - bird_shape_0",
                "method:",
                "  kind: oracle",
            ]
        ),
        encoding="utf-8",
    )

    manifest = run_llm_counterfactual_clue_probe(
        config_path=config_path,
        out_dir=tmp_path / "llm_probe",
        card_top_k=2,
        max_packets=3,
    )

    for key in (
        "cards",
        "feature_clues",
        "latent_clue_packets",
        "hypotheses",
        "test_specs",
        "test_results",
        "training_traces",
        "llm_clues",
        "scores_llm_tested",
        "scores_stats",
        "scores_random",
        "baseline_comparison",
        "manifest",
    ):
        assert Path(manifest[key]).exists()
    assert Path(manifest["latent_clue_packets"]).read_text(encoding="utf-8").strip()
    assert Path(manifest["training_traces"]).read_text(encoding="utf-8").strip()
    assert "test_passed_control" in Path(manifest["llm_clues"]).read_text(encoding="utf-8")
    comparison = Path(manifest["baseline_comparison"]).read_text(encoding="utf-8")
    assert "llm_tested" in comparison
    assert "stats" in comparison
    assert "random" in comparison


def test_bridge_ranker_evaluates_heldout_packet_candidates(tmp_path: Path) -> None:
    root = tmp_path / "bridge_runs"
    for name, good_feature in (("train_a", "feature_good"), ("heldout_b", "feature_good")):
        run_dir = root / name
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            f'{{"config": "configs/experiments/{name}.yaml"}}',
            encoding="utf-8",
        )
        (run_dir / "training_traces.jsonl").write_text(
            "\n".join(
                [
                    '{"feature_name":"feature_good","label_corr":0.9,"env_corr":0.1,"corr_margin":0.8,"abs_corr_margin":0.8,"uncertainty":0.1,"top_group_entropy":0.2,"label_env_disentanglement":0.8,"test_value":1.0,"score_delta":0.5,"hypothesis_correct":true,"passed_control":true}',
                    '{"feature_name":"feature_bad","label_corr":0.1,"env_corr":0.8,"corr_margin":-0.7,"abs_corr_margin":0.7,"uncertainty":0.9,"top_group_entropy":0.8,"label_env_disentanglement":0.1,"test_value":0.0,"score_delta":0.0,"hypothesis_correct":false,"passed_control":false}',
                ]
            ),
            encoding="utf-8",
        )
        (run_dir / "latent_clue_packets.jsonl").write_text(
            "\n".join(
                [
                    '{"feature_index":0,"feature_name":"feature_good","label_corr":0.9,"env_corr":0.1,"corr_margin":0.8,"abs_corr_margin":0.8,"uncertainty":0.1,"top_group_entropy":0.2,"label_env_disentanglement":0.8,"causal_target":1.0}',
                    '{"feature_index":1,"feature_name":"feature_bad","label_corr":0.1,"env_corr":0.8,"corr_margin":-0.7,"abs_corr_margin":0.7,"uncertainty":0.9,"top_group_entropy":0.8,"label_env_disentanglement":0.1,"causal_target":0.0}',
                ]
            ),
            encoding="utf-8",
        )
        (run_dir / "feature_clues.csv").write_text(
            "feature_name,causal_target\nfeature_good,1.0\nfeature_bad,0.0\n",
            encoding="utf-8",
        )

    summary = run_bridge_ranker(
        input_dir=root,
        output_csv=tmp_path / "ranker.csv",
        output_json=tmp_path / "ranker.json",
        top_k_values=[1],
    )

    assert (tmp_path / "ranker.csv").exists()
    bridge = next(row for row in summary["by_label_top_k"] if row["label"] == "bridge_ranker")
    assert bridge["mean_causal_target"] == 1.0


def test_offline_clue_policy_evaluates_heldout_packet_candidates(tmp_path: Path) -> None:
    root = tmp_path / "policy_runs"
    for name in ("train_a", "heldout_b"):
        run_dir = root / name
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            f'{{"config": "configs/experiments/{name}.yaml"}}',
            encoding="utf-8",
        )
        (run_dir / "training_traces.jsonl").write_text(
            "\n".join(
                [
                    '{"candidate_id":"good","feature_name":"feature_good","action":"feature_mean_ablation","label_corr":0.9,"env_corr":0.1,"corr_margin":0.8,"abs_corr_margin":0.8,"uncertainty":0.1,"top_group_entropy":0.2,"label_env_disentanglement":0.8,"test_value":1.0,"score_delta":0.5,"hypothesis_correct":true,"passed_control":true}',
                    '{"candidate_id":"bad","feature_name":"feature_bad","action":"donor_swap_same_label_diff_env","label_corr":0.1,"env_corr":0.8,"corr_margin":-0.7,"abs_corr_margin":0.7,"uncertainty":0.9,"top_group_entropy":0.8,"label_env_disentanglement":0.1,"test_value":0.0,"score_delta":0.0,"hypothesis_correct":false,"passed_control":false}',
                ]
            ),
            encoding="utf-8",
        )
        (run_dir / "latent_clue_packets.jsonl").write_text(
            "\n".join(
                [
                    '{"candidate_id":"good","feature_index":0,"feature_name":"feature_good","label_corr":0.9,"env_corr":0.1,"corr_margin":0.8,"abs_corr_margin":0.8,"uncertainty":0.1,"top_group_entropy":0.2,"label_env_disentanglement":0.8,"causal_target":1.0}',
                    '{"candidate_id":"bad","feature_index":1,"feature_name":"feature_bad","label_corr":0.1,"env_corr":0.8,"corr_margin":-0.7,"abs_corr_margin":0.7,"uncertainty":0.9,"top_group_entropy":0.8,"label_env_disentanglement":0.1,"causal_target":0.0}',
                ]
            ),
            encoding="utf-8",
        )
        (run_dir / "feature_clues.csv").write_text(
            "feature_name,causal_target\nfeature_good,1.0\nfeature_bad,0.0\n",
            encoding="utf-8",
        )

    summary = run_clue_policy_training(
        input_dir=root,
        reward_csv=tmp_path / "rewards.csv",
        output_csv=tmp_path / "policy.csv",
        output_json=tmp_path / "policy.json",
        top_k_values=[1],
    )

    assert (tmp_path / "rewards.csv").exists()
    policy = next(row for row in summary["by_label_top_k"] if row["label"] == "offline_clue_policy")
    fused = next(row for row in summary["by_label_top_k"] if row["label"] == "policy_stats_fused_w0.3")
    safe = next(row for row in summary["by_label_top_k"] if row["label"] == "policy_stats_safe_residual_w0.5")
    assert policy["mean_causal_target"] == 1.0
    assert fused["mean_causal_target"] == 1.0
    assert safe["mean_causal_target"] == 1.0

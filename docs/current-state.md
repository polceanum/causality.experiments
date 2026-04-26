# Current State and Plan

This is the research handoff/state file. Update it after meaningful
implementation or experiment rounds. Keep it focused on the causal-extraction
goal, not incidental development mechanics.

## Operating Constraints

- All core experiments must run locally on the Mac notebook using the existing
  `orpheus` conda environment.
- Do not depend on paid LLM APIs, hosted model APIs, cloud jobs, or external
  processing services.
- Do not replace or pip-install over the Mac-specific PyTorch already present in
  `orpheus`.
- Any real benchmark adapter must retain a tiny local fixture and a modest
  smoke-test path.
- Prefer methods that can produce useful research signal through small models,
  controlled fixtures, and seed sweeps within local compute limits.
- CI should remain a lightweight regression guardrail, not a full experiment
  runner.

## Current State

- Research target:
  surpass published state-of-the-art on at least one real, literature-aligned
  benchmark under matching local-compute assumptions. Fixture wins are only
  development signals.
- `main` is pushed to GitHub.
- Latest pushed commit before the current working round:
  - `1deb2c1` `Align benchmark reporting with literature`
- The repo contains a runnable PyTorch experiment harness for all 8 paper
  experiments using tiny generated fixtures.
- Runnable methods:
  - `constant`
  - `oracle`
  - `erm`
  - `group_balanced_erm`
  - `group_dro`
  - `irm`
  - `jtt`
  - `adversarial_probe`
  - `counterfactual_adversarial`
  - `counterfactual_augmentation`
- Sequence fixtures now use integer tokens, an embedding-pooling classifier,
  and token-specific counterfactual augmentation.
- Seed-sweep scripts are available and have been run on `07_text_toy` and
  `08_fewshot_ner`; robust methods improve mean WGA but with high variance.
- Group-balanced ERM and GroupDRO are implemented as local PyTorch baselines.
- Group-balanced ERM is now a strong baseline on known-group fixtures and must
  be included in future claims.
- JTT is implemented as a local two-stage baseline. It is strong on the
  Waterbirds-style fixture but not uniformly strong on sequence fixtures.
- Hidden-representation extraction and linear causal/nuisance probe diagnostics
  are implemented for small local models.
- Adversarial probe training is implemented. It is promising on Waterbirds-style
  fixtures but currently ineffective on text-toy.
- Counterfactual adversarial training is implemented. It combines nuisance
  counterfactual augmentation, factual/counterfactual consistency, and
  gradient-reversal environment suppression.
- Adapter stubs still intentionally remain for heavier methods:
  causal probes, beta-VAE/iVAE, CITRIS, CSML, and DeepIV.
- Core commands:
  - `conda run -n orpheus python -m pytest`
  - `conda run -n orpheus python scripts/run_all_fixtures.py`
  - `conda run -n orpheus python scripts/run_method_sweep.py`
  - `conda run -n orpheus python scripts/report_best_methods.py`
  - `conda run -n orpheus python scripts/write_research_report.py`
  - `conda run -n orpheus python scripts/run_seed_sweep.py --match 07 --seeds 11,12,13`
  - `conda run -n orpheus python scripts/report_seed_sweep.py --match 07`
  - `conda run -n orpheus python scripts/report_probe_diagnostics.py --match 05_waterbirds`
  - `conda run -n orpheus python scripts/report_benchmark_alignment.py`
  - `conda run -n orpheus python -m causality_experiments run --config configs/benchmarks/waterbirds_features.yaml`
  - `conda run -n orpheus python scripts/run_method_sweep.py --config configs/benchmarks/waterbirds_features.yaml --skip-incompatible --dry-run`
  - `conda run -n orpheus python scripts/run_method_sweep.py --config configs/benchmarks/waterbirds_features.yaml --skip-incompatible`
  - `conda run -n orpheus python -m causality_experiments summarize --runs outputs/runs`
- GitHub Actions CI runs `pytest` plus a tiny CLI smoke test on pushes and pull
  requests.
- Result reports should include literature context where possible. Tiny fixture
  results are not SOTA claims; real benchmark adapters are required for direct
  comparison to published numbers.
- Experiment configs include explicit benchmark metadata marking whether a run
  is a synthetic/local fixture or a real literature-comparable benchmark.
- A local Waterbirds feature-table adapter is available as the first
  literature-aligned benchmark path.
- Benchmark reporting now has a direct comparison view. The alignment report
  shows our latest benchmark-aligned rows next to Waterbirds ERM, JTT,
  GroupDRO, and DFR references, plus delta-to-reference columns and explicit
  statuses such as `fixture_only` and `blocked_missing_local_data`.
- The research report now includes explicit Waterbirds gate-control and compact
  gate-mechanism sections so discovery-vs-random comparisons are visible in one
  generated artifact.
- The repo now also has a compact-to-full validation artifact:
  `outputs/runs/waterbirds-compact-promotion-alignment.csv`, generated by
  `scripts/report_compact_promotion_alignment.py`.
- The real-benchmark status now also checks benchmark provenance. A local CSV is
  not enough by itself; the Waterbirds config must document feature extractor,
  feature source, and split semantics before the repo treats the comparison as
  benchmark-ready.
- The markdown research report now separates blocked real benchmark configs,
  real literature-comparable runs, and literature-aligned fixture runs. The
  current Waterbirds fixture gets a development-only per-method delta table
  against the published references.
- Waterbirds gate mechanism status:
  - fixed gated discovery top-128 remains the best local gate result at test
    WGA about 0.790.
  - fixed instability-JTT plus discovery top-128 reached about 0.713 test WGA,
    so its compact win did not survive promotion.
  - grouped discovery top-128 reaches about 0.752 test WGA.
  - grouped instability-JTT top-128 reaches about 0.785 test WGA and about
    0.797 val WGA.
  - grouped random top-128 reaches about 0.766 test WGA, so grouped discovery
    does not beat a matched random-mask control, but grouped instability-JTT
    now does.
  - recent compact mechanism variants including conditioned, contextual,
    representation-conditioned, disagreement-weighted, and instability-replay
    grouped methods all failed to beat the plain grouped discovery compact
    baseline.
  - grouped instability-JTT is the first grouped mechanism variant that beat
    the compact grouped discovery baseline and survived promotion to a full run.
  - fixed instability-JTT is not the next anchor; grouped instability-JTT is
    the current best follow-on mechanism after fixed discovery.
  - a compact sweep around grouped instability-JTT found a stronger compact
    setting at stage1 epochs 20, top fraction 0.15, upweight 3.0, but its full
    promoted run fell to about 0.673 test WGA. Compact ranking is therefore not
    sufficient to reorder the full-run leaderboard by itself.
  - compact promotion should now require both compact test WGA and compact val
    WGA to clear a threshold with only a small test-val gap; test-only compact
    wins are not enough.
  - a stability-penalized stage-1 score (`mean_minus_std`) was added as a
    principled follow-up, but a compact sweep around that variant produced no
    promotable candidates and stayed below the current grouped instability-JTT
    compact baseline.
  - a loss-weighted stage-1 score (`loss_weighted_mean`) improved on the
    stability-penalized follow-up but still produced no promotable compact
    candidates; its best compact promotion score was about 0.639, still below
    the grouped instability-JTT compact bar.
  - a counterfactual excess-loss stage-1 score
    (`counterfactual_loss_increase_mean`) reached about 0.667 compact test WGA
    but only about 0.647 compact val WGA, so it also failed promotion.
  - a group-weighted counterfactual excess-loss stage-1 score
    (`group_loss_weighted_counterfactual_loss_increase_mean`) slightly improved
    compact test WGA to about 0.673 but fell to about 0.639 compact val WGA,
    so it also failed promotion and did not improve promotion score.
  - the stage-1 selector branch now looks exhausted locally; the next
    principled step is to change the stage-2 objective itself, not keep adding
    more selector-weighting variants.

## Near-Term Plan

1. Get a real Waterbirds-compatible feature run.
   - Use local features, real splits, labels, and group/background metadata.
   - Compare WGA against literature references in `docs/literature-context.md`.
   - If the feature table has known bird-specific feature columns, set
     `dataset.causal_feature_columns` or `dataset.causal_feature_prefixes` so
     counterfactual methods can run honestly.
2. Compose the strongest local mechanisms.
   - Future mechanism changes should be screened first against the matched
     fixed/grouped random-mask controls, not just against earlier discovery
     variants.
   - Use the stricter compact promotion rule before any full-budget promotion:
     clear compact test WGA, compact val WGA, and a small test-val gap.
   - Use grouped instability-JTT as the new grouped-family comparator.
   - Stage-1 signal changes should be judged by promotion score, not raw
     compact test WGA, since both stable and loss-weighted follow-ups failed
     promotion despite being mechanistically plausible.
   - Stop extending the stage-1 selector family by default; the loss-delta and
     group-loss-delta selectors also failed promotion, so the next mechanism
     work should target the stage-2 counterfactual-adversarial objective.
   - Counterfactual-adversarial improvements should still clear the current
     fixed/grouped random controls before being promoted to full research
     claims.
3. Develop factor/token-specific probe interventions.
   - Generic adversarial hiding is too blunt for sequence fixtures.
   - Use known factor/token metadata to design targeted intervention losses.
4. Use group-balanced ERM as a required comparator for any known-group result.
5. For every serious result, compare against literature reference/SOTA numbers
   and cite the source; update `docs/literature-context.md` as needed.

## Current Risks

- Fixture tasks are useful for harness iteration but are not substitutes for
  real benchmark results.
- Current sequence experiments are under-modeled.
- IRM remains sensitive to penalty schedules and may need config sweeps per
  dataset.
- ATE proxy is a rough diagnostic, not a validated causal-effect estimator.
- Sequence WGA improvements are promising but high-variance across seeds.
- Some known-group baselines improve WGA by sacrificing average accuracy; reports
  should continue showing both.
- Fixture wins can be misleadingly high; compare to SOTA only on real benchmark
  adapters with matching assumptions.
- The local Waterbirds feature table now exists, but compact screening remains
  an imperfect proxy for full-budget ordering; promotion decisions still need
  explicit test/validation thresholds and follow-up full runs.
- Strong matched random-mask controls remain a real risk to overclaiming; any
  future mechanism win still needs to clear the current fixed/grouped random
  baselines before it is treated as causal progress.

## Log Hygiene

- `docs/research-log.md` should track research progress and empirical signals.
- `docs/failed-attempts.md` should track failed or weak research/modeling
  attempts, not tooling friction.
- `docs/current-state.md` should stay as a compact handoff: current capability,
  next plan, risks, and active assumptions.

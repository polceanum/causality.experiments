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

## Near-Term Plan

1. Get a real Waterbirds-compatible feature run.
   - Use local features, real splits, labels, and group/background metadata.
   - Compare WGA against literature references in `docs/literature-context.md`.
2. Compose the strongest local mechanisms.
   - Tune counterfactual adversarial training against JTT and group-balanced
     ERM before claiming progress.
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

## Log Hygiene

- `docs/research-log.md` should track research progress and empirical signals.
- `docs/failed-attempts.md` should track failed or weak research/modeling
  attempts, not tooling friction.
- `docs/current-state.md` should stay as a compact handoff: current capability,
  next plan, risks, and active assumptions.

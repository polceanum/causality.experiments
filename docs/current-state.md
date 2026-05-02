# Current State and Plan

This is the compact research handoff. Keep detailed chronology in
`docs/research-log.md`, and keep weak or failed paths in
`docs/failed-attempts.md`. This file should answer: what is currently true,
what is the bar, what should be tried next, and what should not be repeated.

## Operating Constraints

- Run core work locally on the Mac notebook in the existing `orpheus` conda
  environment.
- Do not depend on paid LLM APIs, hosted model APIs, cloud jobs, or external
  processing services.
- Do not replace or pip-install over the Mac-specific PyTorch already present in
  `orpheus`.
- Keep CI lightweight: regression tests plus tiny smoke paths, not full
  experiments.
- Every real benchmark adapter needs a tiny local fixture and a modest smoke
  path.
- Treat fixture wins as development signals only. Literature-facing claims need
  real benchmark adapters, matching assumptions, seed checks, and published
  reference context.

## Current Snapshot

- Full local regression suite and GitHub Actions are green at `146` tests.
- The repo has a runnable PyTorch experiment harness for all eight paper
  fixtures and a local Waterbirds benchmark path.
- Runnable methods include `constant`, `oracle`, `erm`, `dfr`, `causal_dfr`,
  `group_balanced_erm`, `group_dro`, `irm`, `jtt`, `adversarial_probe`,
  `counterfactual_adversarial`, and `counterfactual_augmentation`.
- Adapter stubs remain for heavier methods: causal probes, beta-VAE/iVAE,
  CITRIS, CSML, and DeepIV.
- The active research target is a real, literature-aligned benchmark result
  that beats the relevant local comparator under matching local-compute
  assumptions. The main benchmark-facing target remains Waterbirds.

## Core Commands

```bash
conda run -n orpheus python -m pytest
conda run -n orpheus python scripts/run_all_fixtures.py
conda run -n orpheus python scripts/run_method_sweep.py
conda run -n orpheus python scripts/report_best_methods.py
conda run -n orpheus python scripts/write_research_report.py
conda run -n orpheus python -m causality_experiments summarize --runs outputs/runs
```

Waterbirds and clue-specific entry points:

```bash
conda run -n orpheus python scripts/report_benchmark_alignment.py
conda run -n orpheus python scripts/run_waterbirds_official_dfr.py
conda run -n orpheus python scripts/run_waterbirds_official_causal_dfr_sweep.py
conda run -n orpheus python scripts/run_waterbirds_official_representation_sweep.py
conda run -n orpheus python scripts/run_waterbirds_official_backbone_sweep.py
conda run -n orpheus python scripts/run_waterbirds_clue_seed_stability.py
conda run -n orpheus python scripts/run_llm_counterfactual_clue_probe.py --config configs/experiments/01_synthetic_linear.yaml --llm-backend mock
```

## Benchmark Bar

### Waterbirds Comparator

- The official-aligned local comparator is
  `official_dfr_val_tr_retrains50` on
  `data/waterbirds/features_official_erm_official_repro.csv`.
- Its current local test WGA is about `0.933`. This is the bar for Track A
  Waterbirds mechanism claims.
- Against the currently recorded literature SOTA/reference DFR WGA `0.929`, the
  local 50-retrain official comparator is about `+0.004` WGA. Treat this as a
  validated reproduction/comparator, not as a proposed-method improvement.
- The legacy `waterbirds_features_dfr` config still runs on
  `data/waterbirds/features.csv`, but its latest local test WGA is about
  `0.897`, so it is a historical anchor rather than the benchmark-facing bar.
- The older 20-retrain comparator at about `0.931` is useful for historical
  continuity, but should not be the promotion bar.
- DFR validation metrics are protocol diagnostics, not unbiased holdout metrics,
  because final classifiers are trained on validation groups.

### Promotion Rule

Promote a Waterbirds candidate only if it satisfies all of these:

- It is seed-matched against the active official comparator.
- Mean test WGA is above the comparator by more than noise.
- Variance and minimum paired seed delta are acceptable.
- It clears matched random/control baselines when masks or scores are involved.
- The result uses real benchmark features and documented provenance, not only a
  tiny fixture or limit-slice artifact.

Single-seed wins, compact-only wins, and validation-selected wins are diagnostic
signals, not research claims.

## Active Tracks

### Track A: Official-Feature Mechanisms

Goal: beat the official DFR comparator without changing the base feature table.

Current state:

- Plain official DFR reproduction works locally and slightly exceeds the
  literature reference, so the earlier protocol gap is closed.
- `causal_dfr` has single-seed upside but fails seed-matched stability. A
  seeds `101`-`103` smoke averaged about `0.9252` test WGA, with mean paired
  delta about `-0.0062` against the official baseline.
- `official_causal_shrink_dfr_val_tr` is implemented and exactly recovers
  official DFR at shrink `1.0`, but gentle hard/soft shrink grids did not beat
  the 50-retrain comparator.
- A paired official-shrink sweep runner is now available at
  `scripts/run_waterbirds_official_shrink_sweep.py`. The first near-identity
  shrink screen against `official_dfr_val_tr_retrains50` tied at best on
  seed 101 and did not produce target-beating evidence.
- Feature-swap `counterfactual_adversarial` is non-competitive on official
  features. The swap construction appears too off-manifold for deep Waterbirds
  representations.
- `official_representation_dfr` is implemented. Current adversarial/gated
  representation probes improve diagnostics such as nuisance-to-causal
  importance, but still underperform test WGA.

Decision:

- Do not spend more compute on broad unpaired `causal_dfr` or shrink grids.
- Do not keep extending near-identity final-head shrink unless a better upstream
  score/support source first creates clear held-out signal.
- Any next Track A run should change the mechanism qualitatively and report
  paired deltas against `official_dfr_val_tr_retrains50`.

### Track B: Feature Generation and Representation Source

Goal: improve the representation before the unchanged official DFR head.

Current state:

- The official backbone sweep has provenance guardrails. Feature manifests must
  record resolved ERM/backbone settings, and rows with missing settings or
  collapsed base ERM WGA are blocked before DFR scoring.
- The prior broken Track B feature artifact is formally blocked: one cached
  artifact lacked resolved settings, and a fresh no-env-adv rerun reproduced a
  real base-ERM collapse.
- CPU is now the default for backbone sweeps unless `--device auto` is
  requested, because local MPS/Metal failures occurred during long reruns.
- The first no-limit CPU candidate, e5/LR `0.001`/no-env-adv/seed `101`,
  cleared guardrails but reached only about `0.9143` downstream official DFR
  test WGA. Do not seed-sweep or promote it.
- Limit384 follow-ups with group balancing, env-adv, staged conflict sampling,
  and global supervised contrastive objectives were negative. Several improved
  base ERM WGA while worsening downstream DFR WGA, so base classifier strength
  alone is not the right selection signal.
- Frozen stronger-source diagnostics were negative on limit384: ResNet50-V2,
  ConvNeXt-Tiny, CLIP ViT-B/32, and DINOv2-small did not justify full runs.
- Component/decomposed feature export is available. DINO patch/crop components
  looked promising on limit384 but failed to beat the comparator in the first
  no-limit patch-component run.

Decision:

- Do not launch full runs for the exact failed e5, group-balanced e5, env-adv
  e5, staged conflict, global contrastive, or frozen-source settings.
- Future Track B work needs a more structured representation intervention than
  global sampling/contrast alone. If revisiting DINO decomposition, first make
  extraction efficient and checkpointed enough for full-data runs.

### LLM-Guided Clue Bridge

Goal: use a local/replayable planner to propose causal explanations and cheap
feature-level tests, then train a bridge that predicts which probes are worth
running or trusting.

Current implementation:

- `causality_experiments.latent_clue_packets` converts feature/probe evidence
  into stable latent clue packets.
- `causality_experiments.llm_clue_planner` provides a JSON-constrained mock
  planner over a fixed action catalog.
- `causality_experiments.counterfactual_clue_tests` executes deterministic
  ablation, shrink, donor-swap, and conditional-signal checks with matched
  random-feature controls.
- `causality_experiments.llm_clue_bridge` turns packet/plan/result traces into
  training targets: `hypothesis_label`, `test_value`, and `score_delta`.
- `scripts/run_llm_counterfactual_clue_probe.py` writes replayable packets,
  hypotheses, test specs, measured test results, tested clue rows, bridge
  traces, and baseline comparison artifacts.

Fixture comparison:

- Across all eight lightweight fixtures, `llm_tested` top-1 known causal-target
  recovery averaged `0.625`, versus `1.000` for the stats baseline and `0.000`
  for deterministic random.
- At top-2, `llm_tested` averaged `0.3125`, versus `0.6875` for stats and
  `0.125` for random.
- Interpretation: the tested loop is extracting non-random information, but the
  simple stats baseline is stronger. Misses concentrate on factor/sequence
  fixtures where observed effects or correlations can be shortcut/artifact
  clues rather than latent causes.

Decision:

- Treat these misses as bridge-training data, not as a reason to hard-code the
  mock planner toward fixture ground truth.
- Next implementation layer: add a replay backend, train/evaluate a small
  ranker or contextual bandit on packet/plan/result traces, and hold out
  fixture families to test generalization.
- First held-out bridge ranker is implemented at
  `scripts/train_llm_clue_bridge_ranker.py`. On existing fixture artifacts it
  beats stats-margin on candidate-packet causal-target recovery: top-1 bridge
  `0.625` versus stats `0.500`, top-2 bridge `0.4375` versus stats `0.3125`,
  and top-4 bridge `0.34375` versus stats `0.250`.
- Bridge scores now feed the Waterbirds clue-fusion path as opt-in `bridge` and
  `bridge_fused` sources. The conservative `bridge_fused` top-512 candidate
  reached test WGA `0.9345794320` with 50 retrains, versus a fresh locked
  official DFR comparator rerun at `0.9330217838` and a matching stats top-512
  control at `0.9314641953`.
- Treat this as the first downstream bridge win, not yet a benchmark claim. It
  is a seed-101 improvement of about `+0.00156`, below the current promotion
  gate. Next step is paired multi-seed bridge-fused screening before changing
  the benchmark headline.

### Clue Fusion and Discovery Masks

Current state:

- The clue-fusion bridge creates feature cards, deterministic language clues,
  image/prototype activation clues, stats/language/image/fused score CSVs, and
  source ablation reports.
- On official Waterbirds repro features, language clues provide non-neutral weak
  evidence from activation alignment rather than feature names. Language-only
  top-k sets differ from stats-only sets, and fused scores inject label-aligned
  confidence while preserving some statistical margin.
- Existing downstream consumers have not produced a stable Waterbirds win.
  Soft-score official shrink almost tied the comparator but did not clear the
  promotion gate, and stronger `causal_dfr` soft-score objectives failed paired
  seed checks.
- `dfr_num_retrains` makes soft-score ensembles more stable, but not stronger
  than official DFR.

Decision:

- Keep clue fusion as a diagnostic and as supervision for the LLM bridge.
- Do not promote score/mask consumers unless they beat matched stats/random and
  official DFR controls under paired seed checks.

### Gate Mechanisms

Current state:

- Fixed gated discovery top-128 remains the best local fixed-gate result at
  about `0.790` test WGA.
- Grouped instability-JTT is the best grouped follow-on mechanism, but compact
  improvements did not reliably survive full promotion.
- Recent selector variants, including stability-penalized, loss-weighted,
  counterfactual excess-loss, and group-weighted counterfactual excess-loss,
  did not produce promotable candidates.

Decision:

- Stop extending stage-1 selector weighting by default.
- If returning to this family, change the stage-2 objective and clear matched
  fixed/grouped random-mask controls before any claim.

## Retired or Low-Priority Paths

- The active latent patch-probe runner was pruned. It improved logit-drop
  diagnostics but did not produce seed-stable downstream WGA, and the only
  compact bump was a one-example/C-grid artifact. Preserve the lesson in
  `docs/failed-attempts.md`; do not restart from that runner.
- Fixed-feature DFR head tweaks, validation-threshold calibration, LBFGS
  logistic DFR, exact balanced validation subsampling, and train-feature
  standardization did not close the Waterbirds gap.
- Layer4-only ERM feature ladders, plain full-backbone SGD plus augmentation,
  and small group-weight/sample-weight variations did not produce a promotable
  feature source.
- Generic adversarial hiding remains too blunt for sequence fixtures; any
  sequence work should be factor/token-specific.

## Near-Term Plan

1. Build the trainable clue bridge.
   - Add a replay backend for planner outputs.
   - Train a small ranker/contextual bandit over latent packets, proposed
     actions, and measured test results.
   - Evaluate against stats and deterministic random on held-out fixture
     families before touching Waterbirds downstream consumers.

2. Keep Track A honest.
   - Use `official_dfr_val_tr_retrains50` as the paired comparator.
   - Run only mechanism changes that are qualitatively different from the
     exhausted causal-DFR/shrink/swap grids.
   - Report mean paired delta, minimum paired delta, and variance.

3. Revisit representation generation only with structure.
   - Avoid full runs for the failed global sampling/contrast recipes.
   - If using component features again, make extraction efficient,
     checkpointed, and provenance-audited first.
   - Prefer interventions that preserve foreground/bird signal while reducing
     background shortcut reliance.

4. Keep benchmark reporting current.
   - Compare serious Waterbirds rows against `docs/literature-context.md`.
   - Keep `docs/research-log.md` updated after meaningful experiment rounds.
   - Move failed or weak paths into `docs/failed-attempts.md` once the lesson is
     clear.

## Current Risks

- Fixture tasks can make weak mechanisms look strong; real benchmark adapters
  remain the only route to literature claims.
- Compact screens are useful but do not reliably order full-budget runs.
- Strong matched random controls are still a major overclaiming risk for masks,
  scores, and clue-derived feature selection.
- Waterbirds DFR validation metrics are not unbiased holdout metrics.
- Better base ERM WGA does not necessarily imply better downstream DFR WGA.
- Current sequence experiments are under-modeled, and WGA improvements there
  remain high variance across seeds.

## Log Hygiene

- `docs/research-log.md`: chronological research progress and empirical
  signals.
- `docs/failed-attempts.md`: failed or weak research/modeling attempts and the
  lesson to preserve.
- `docs/current-state.md`: compact handoff, active assumptions, promotion bars,
  next steps, and stop-rules.

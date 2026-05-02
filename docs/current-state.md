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

- Full local regression suite is green at `159 passed` after the bridge
  candidate reporting and support-audit tooling addition.
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
conda run -n orpheus python scripts/train_llm_clue_policy.py --input-dir outputs/dfr_sweeps/llm_clue_fixture_experiments_20260502_refreshed
conda run -n orpheus python scripts/report_waterbirds_bridge_candidate.py
conda run -n orpheus python scripts/report_waterbirds_bridge_support.py
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
- `scripts/run_llm_clue_fixture_experiments.py` regenerates the eight-fixture
  replay corpus with explicit `card_top_k`/`max_packets` settings; the refreshed
  Waterbirds bridge result uses its default `max_packets: 16` corpus shape.

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
  `scripts/train_llm_clue_bridge_ranker.py`. Its historical fixture-corpus
  result showed bridge top-1 causal-target recovery `0.625` versus stats
  `0.500`; after refreshing the ignored fixture traces, the same evaluation
  became bridge `0.25` versus stats `0.625`. Treat fixture trace provenance as
  part of the experimental state, and do not compare refreshed-trace results to
  the older held-out bridge number as if they were the same training corpus.
- Bridge scores feed the Waterbirds clue-fusion path as opt-in `bridge`,
  `bridge_fused`, and `bridge_gated` sources. The old `bridge_fused/w0.2/top512`
  two-seed result was positive but small; after refreshing the replayed fixture
  traces, `w0.2/top512` no longer beat the 50-retrain comparator on seed 101.
- The refreshed-trace active candidate is `bridge_fused/w0.3/top512` through the
  official causal-shrink DFR consumer. Five full 50-retrain paired seeds
  `101`-`105` give mean candidate WGA `0.9367601395`, min WGA `0.9361370802`,
  mean paired delta `+0.0062305570` to the seed-matched official DFR baseline,
  and mean delta `+0.0031152844` to the stats top-512 control. All five seeds
  are positive against the official baseline and non-negative against stats.
- Matched random-score controls now run through the same discovery-score,
  pruned-soft-score, official-shrink consumer path. Across three deterministic
  random score controls and the same five seeds, the best random-control mean
  WGA was `0.9317757010`; the bridge candidate beat the best random control on
  mean by `+0.0049844384` WGA and was non-negative against the best random
  control on every seed, with mean paired delta `+0.0021807075`.
- The refreshed fixture trace corpus used for this result is snapshotted under
  `outputs/dfr_sweeps/llm_clue_fixture_experiments_20260502_refreshed`, and a
  seed-101 sanity rerun from that path reproduced candidate WGA `0.9376947284`.
  Treat `bridge_fused/w0.3/top512` as the new active Track A candidate. It has
  crossed the local promotion gap against official DFR, stats, and matched
  random controls; the next audit step is to make the trace snapshot or score
  artifact part of a durable benchmark manifest.
- `scripts/report_waterbirds_bridge_candidate.py` now turns the paired CSV,
  score files, and trace snapshot into a checksum-backed promotion report. The
  generated report at `outputs/dfr_sweeps/bridge-fused-candidate-report.md`
  recomputes the candidate-vs-best-random gate from raw rows and records score
  CSV SHA256 hashes.
- `scripts/report_waterbirds_bridge_support.py` now summarizes the selected
  top-k support behind the incumbent. On the current top-512 artifacts,
  `bridge_fused/w0.3` overlaps stats on `311/512` features (`0.4362` Jaccard),
  while deterministic random controls overlap only `124`-`145` features
  (`0.1378`-`0.1650` Jaccard). Bridge-fused selects only `5` features where
  `env_corr >= label_corr`, versus `91`-`93` for random controls.
- `scripts/run_waterbirds_bridge_fused_sweep.py` now supports opt-in bridge
  support variants for controlled screens: `env_filter`, `margin_gate`,
  `stats_fill`, `soft_env_penalty`, `stats_anchor`, `score_sqrt`, and
  `score_square`. These are diagnostics and candidate generators, not new
  defaults.
- The first support-composition iteration did not widen the full-budget margin.
  Hard `env_filter` looked slightly better in a two-seed, five-retrain compact
  screen, but the five-seed 50-retrain gate underperformed the incumbent:
  mean WGA `0.9361370683` versus incumbent `0.9367601395`, with only `3/5`
  non-negative deltas against the best random control. Softer penalties and
  monotone score-shape transforms did not create a larger compact margin.
- The first RL/contextual-bandit slice is implemented in
  `causality_experiments.rl_clue_policy` and
  `scripts/train_llm_clue_policy.py`. It builds explicit offline reward rows
  from packet/trace/clue artifacts, marks reward scope and trainability, blocks
  benchmark-final/test reward rows from training, and evaluates a ridge value
  policy leave-one-fixture-out. On the refreshed trace snapshot with alpha
  `10`, raw policy top-1 causal recovery is weak (`0.25` versus stats `0.625`),
  but conservative normalized policy/stat fusion at `w=0.3` preserves stats
  top-1 (`0.625`) and improves top-2/top-4 recovery (`0.375`/`0.34375` versus
  stats `0.3125`/`0.28125`). Treat the fused policy as a development signal,
  not a Waterbirds promotion candidate until it clears a downstream compact
  screen.

### Clue Fusion and Discovery Masks

Current state:

- The clue-fusion bridge creates feature cards, deterministic language clues,
  image/prototype activation clues, stats/language/image/fused score CSVs, and
  source ablation reports.
- On official Waterbirds repro features, language clues provide non-neutral weak
  evidence from activation alignment rather than feature names. Language-only
  top-k sets differ from stats-only sets, and fused scores inject label-aligned
  confidence while preserving some statistical margin.
- The refreshed bridge-fused official-shrink consumer now has a five-seed
  positive Waterbirds signal, while stronger validation-split `causal_dfr`
  soft-score objectives still fail paired seed checks.
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

1. Freeze the active bridge result as a durable benchmark artifact.
   - Keep using the snapshotted trace directory, score CSV hashes, paired rows,
     stats control, and deterministic random-score controls.
   - Treat mutable ignored output directories as insufficient for future claims
     unless a report/manifest records the exact files used.

2. Use support diagnostics to define the next candidate family.
   - The incumbent is not random-like: it shares much more support with stats
     while avoiding the random controls' env-dominant selected features.
   - Simple hard/soft env filtering and monotone score-shape transforms have
     now been screened; do not promote them unless a materially different gate
     changes the objective.
   - The next official-feature improvement should change bridge supervision or
     the official-compatible consumer more substantially than score reshaping.

3. Improve bridge supervision only behind held-out gates.
   - Extend the new offline reward table toward pairwise/listwise ranking or a
     two-stage artifact-risk bridge, but require held-out fixture improvement or
     a compact Waterbirds screen before 50-retrain promotion.
   - Keep raw offline policy below promotion until it beats stats on held-out
     fixture top-1; the current useful signal is only the conservative
     policy/stat fusion.
   - Do not re-add raw activation-gap fields directly without a versioned corpus
     and held-out win.

4. Keep benchmark reporting current.
   - Refresh `docs/literature-context.md` before any benchmark-facing claim.
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

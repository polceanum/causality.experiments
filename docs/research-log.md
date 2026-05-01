# Research Log

Chronological notes about experiments, implementation choices, and empirical
signals. Keep this focused on what was tried and what was learned.

## 2026-04-22

- Read the source document, which is a survey and experiment blueprint rather
  than a single fully specified algorithm. The practical research direction is
  to compare counterfactual interventions, invariance penalties, probing, and
  meta-causal generalization under shared evaluation scaffolding.
- Set an operating constraint for the research program:
  all core experiments must run locally on the Mac notebook in the `orpheus`
  environment, without paid APIs, hosted model APIs, cloud processing, or
  replacing the existing Mac-specific PyTorch install.
- Chose a dataset-driven harness as the first implementation target:
  all 8 paper experiments are represented by runnable tiny fixtures, with
  adapter boundaries for real benchmark data.
- Implemented the first runnable harness:
  dataset adapters, configs, method interface, metrics, CLI, fixture generation,
  and batch runs.
- Initial runnable methods:
  `constant`, `oracle`, and `erm`.
- Initial metrics:
  accuracy, worst-group accuracy, support recovery, and an ATE-style proxy when
  fixture ground truth supports it.
- Ran all 8 fixture configs successfully.
- Added runnable `irm` and `counterfactual_augmentation` methods.
- Ran a 24-run method sweep across all 8 fixture configs with ERM, IRM, and
  counterfactual augmentation.
- Main empirical signal from the first sweep:
  counterfactual augmentation strongly improves worst-group accuracy on the
  clean spurious-feature fixtures, especially synthetic tabular and
  Waterbirds-style data.
- IRM is sensitive to penalty scale:
  the first sweep used `penalty_weight=50.0`, which often collapsed. A quick
  tuning pass showed lower penalties are safer, so the sweep default was changed
  to `penalty_weight=1.0` with longer training.
- Current best-method summary from fixture sweeps:
  - `01_synthetic_linear`: counterfactual augmentation, WGA about 0.94.
  - `02_synthetic_nonlinear`: counterfactual augmentation, WGA about 0.82.
  - `03_dsprites_3dshapes`: counterfactual augmentation, WGA about 0.20.
  - `04_causal3dident`: IRM, WGA about 0.13.
  - `05_waterbirds`: counterfactual augmentation, WGA about 0.93.
  - `06_shapes_spurious`: IRM, WGA about 0.32.
  - `07_text_toy`: ERM remains best, WGA about 0.40.
  - `08_fewshot_ner`: IRM edges out others, WGA about 0.51.
- Interpretation:
  the current tabular treatment is enough to test spurious-feature robustness,
  but sequence/text fixtures need a real sequence model and intervention logic.
- Research conclusion so far:
  explicit counterfactual nuisance randomization is the strongest initial
  approach for simple spurious-correlation tasks, while invariance-only training
  needs careful penalty tuning and task-specific validation. The next research
  bottleneck is making interventions semantically valid for structured data.

## 2026-04-22: Sequence Track Upgrade

- Changed sequence fixtures to preserve integer tokens instead of normalized
  float vectors.
- Added an embedding-pooling classifier for sequence datasets.
- Changed counterfactual augmentation on sequence fixtures to swap only the
  confounder token position while preserving the causal token.
- Focused sweep on the sequence tasks:
  - `07_text_toy`: ERM reached higher average accuracy, but worst-group
    accuracy dropped; IRM and counterfactual augmentation had lower average
    accuracy but better worst-group accuracy.
  - `08_fewshot_ner`: IRM had the best worst-group accuracy among current
    methods, with counterfactual augmentation also improving over ERM.
- Interpretation:
  the sequence model now captures more signal, but ERM still exploits the
  shortcut. Robust methods improve WGA, suggesting the task is now a more useful
  intervention benchmark than the original float-token MLP setup.

## 2026-04-22: Sequence Seed Sweep

- Added seed-sweep scripts so fixture claims can move from single-run anecdotes
  to mean/std comparisons.
- Ran three-seed sweeps on `07_text_toy` and `08_fewshot_ner`.
- `07_text_toy` mean results:
  - ERM: WGA 0.235 +/- 0.098, accuracy 0.763 +/- 0.008.
  - IRM: WGA 0.364 +/- 0.208, accuracy 0.743 +/- 0.028.
  - Counterfactual augmentation: WGA 0.338 +/- 0.078, accuracy 0.763 +/- 0.018.
- `08_fewshot_ner` mean results:
  - ERM: WGA 0.162 +/- 0.193, accuracy 0.725 +/- 0.014.
  - IRM: WGA 0.241 +/- 0.251, accuracy 0.694 +/- 0.037.
  - Counterfactual augmentation: WGA 0.266 +/- 0.215, accuracy 0.708 +/- 0.024.
- Interpretation:
  robust methods improve mean WGA on both sequence fixtures, but variance is
  large. This is not enough to claim a robust sequence method yet; it does
  justify using these fixtures as stress tests while adding stronger baselines
  and better intervention policies.

## 2026-04-22: Known-Group Robust Baselines

- Added local PyTorch implementations of:
  - `group_balanced_erm`: samples minibatches with inverse group-frequency
    weights.
  - `group_dro`: maintains exponential weights over group losses and optimizes
    the weighted group-risk objective.
- Added both baselines to method sweeps, seed sweeps, reports, and regression
  tests.
- Focused single-run sweeps on synthetic linear, Waterbirds-style, text toy, and
  few-shot NER fixtures.
- Main single-run signal:
  - Waterbirds-style: group-balanced ERM reached WGA about 0.91, while
    counterfactual augmentation remained strongest at about 0.93.
  - Text toy: group-balanced ERM and GroupDRO improved WGA over ERM, but with
    lower average accuracy than ERM/counterfactual augmentation.
  - Few-shot NER: group-balanced ERM matched counterfactual augmentation WGA but
    trailed IRM on the focused single run.
- Three-seed Waterbirds-style sweep:
  - counterfactual augmentation and group-balanced ERM both reached WGA/accuracy
    near 1.0 on this fixture.
  - GroupDRO improved WGA relative to ERM but did not match the two strongest
    methods.
  - IRM underperformed on this known-group fixture.
- Sequence seed sweeps with the new baselines:
  - `07_text_toy`: group-balanced ERM had the best mean WGA among tested methods
    but reduced average accuracy.
  - `08_fewshot_ner`: group-balanced ERM and counterfactual augmentation had the
    best mean WGA, again with accuracy trade-offs.
- Interpretation:
  known-group balancing is now a necessary baseline. Future claims must beat
  group-balanced ERM, not just ERM/IRM. Counterfactual augmentation remains
  competitive and sometimes stronger, but its advantage is not universal.

## 2026-04-22: JTT-Style Two-Stage Baseline

- Added a local JTT-style baseline:
  first train ERM, identify misclassified training examples, then retrain with
  those examples upweighted by sampling weight.
- Added `jtt` to method sweeps, seed sweeps, reports, README, and regression
  tests.
- Focused single-run results:
  - Waterbirds-style: JTT reached WGA/accuracy 1.0 on the latest run.
  - Text toy: JTT improved WGA over ERM but with a large average-accuracy
    trade-off.
  - Few-shot NER: JTT underperformed IRM and counterfactual augmentation on WGA.
- Seed-sweep results:
  - Waterbirds-style: JTT had WGA/accuracy 1.0 across the three new seeds,
    matching or slightly exceeding counterfactual augmentation and
    group-balanced ERM on this fixture.
  - Text toy: JTT did not beat group-balanced ERM on mean WGA and had lower
    average accuracy.
- Interpretation:
  JTT is a strong local baseline for clean spurious-correlation settings, but it
  is not a general solution for the sequence fixtures. Future method claims
  should compare against JTT on Waterbirds-style tasks and against
  group-balanced ERM on known-group sequence tasks.

## 2026-04-22: First Causal Probe Diagnostics

- Added hidden-representation extraction for the small MLP and sequence models.
- Added linear probe diagnostics to every evaluation:
  - `probe/causal_accuracy`
  - `probe/nuisance_accuracy`
  - `probe/selectivity`
- Added a probe diagnostics report script.
- Initial Waterbirds-style diagnostic:
  robust methods with high WGA have slightly higher causal-vs-nuisance
  selectivity than ERM/IRM/GroupDRO, but nuisance information remains decodable.
- Initial text-toy diagnostic:
  nuisance tokens remain more decodable than causal tokens across all current
  methods, matching the weak/high-variance sequence WGA story.
- Interpretation:
  probing is useful as a diagnostic, but this version is not yet a causal
  intervention method. The next research step is to use probe information to
  regularize or intervene on representations, then compare against JTT and
  group-balanced ERM.

## 2026-04-22: Adversarial Probe Training

- Added `adversarial_probe`, a local gradient-reversal method that trains the
  classifier while an auxiliary head tries to predict environment labels from
  hidden representations.
- Focused Waterbirds-style result:
  adversarial probe training reached WGA/accuracy 1.0 on the latest run,
  matching JTT and exceeding group-balanced ERM on that single run.
- Adversarial-weight tuning on Waterbirds-style:
  `adv_weight=0.05` preserved WGA/accuracy 1.0 while lowering nuisance probe
  accuracy more than larger weights in the local sweep.
- Text-toy result:
  adversarial probe training did not improve WGA over ERM and did not reduce
  nuisance decodability enough to matter.
- Interpretation:
  adversarial probe training is a promising local mechanism for clean
  spurious-feature settings, but it is not yet robust on sequence fixtures.
  The next version likely needs token/factor-specific interventions rather than
  only hiding environment information from pooled representations.

## 2026-04-22: Literature Context Requirement

- Added a reporting rule:
  future result summaries should compare against existing literature reference
  or SOTA numbers where possible.
- Important caveat:
  current benchmark names like `05_waterbirds` are tiny local fixtures, not real
  benchmark adapters. Their high WGA values are useful for iteration but not
  comparable to published Waterbirds numbers.
- Added `docs/literature-context.md` with initial Waterbirds/JTT context and a
  reminder to refresh sources before serious claims.
- Added benchmark metadata to experiment configs so reports can distinguish
  synthetic/local fixtures from real literature-comparable benchmark runs.
- Added a benchmark-alignment report to show when a run has literature
  references but is still only a fixture result.
- Added a local Waterbirds feature-table adapter and config template as the
  first path toward a real literature-aligned benchmark run without cloud/API
  dependencies.

## 2026-04-22: SOTA Target

- Set the explicit research target:
  surpass published SOTA on at least one real, literature-aligned benchmark
  under matching assumptions and local compute constraints.
- Initial practical target:
  Waterbirds WGA, using local feature-table inputs and comparing against
  published Waterbirds references.
- Guardrail:
  fixture performance, even when very high, is not evidence of SOTA.

## 2026-04-22: Counterfactual Adversarial Composition

- Added `counterfactual_adversarial`, a local method that combines:
  counterfactual nuisance augmentation, factual/counterfactual prediction
  consistency, and gradient-reversal environment suppression.
- Focused Waterbirds-style fixture result:
  `counterfactual_adversarial` reached WGA/accuracy 1.0, matching JTT and
  adversarial probe on this easy fixture.
- Probe diagnostic on the same fixture:
  `counterfactual_adversarial` reached causal probe accuracy 1.0, nuisance
  probe accuracy about 0.869, and selectivity about 0.131. This is better
  selectivity than plain adversarial probe in the latest local run, but it is
  still only a fixture result.
- Interpretation:
  composing counterfactual augmentation with adversarial suppression is a
  plausible SOTA-seeking direction, but the current evidence is development
  signal only. The next decisive step is to run the same method on a real
  Waterbirds-compatible local feature table and compare against published WGA
  references.

## 2026-04-22: Real-Adapter Readiness

- Checked for `data/waterbirds/features.csv`; no local feature table is present
  yet, so no real Waterbirds comparison can be run on this machine right now.
- Added an optional causal-mask contract to the Waterbirds feature adapter:
  `dataset.causal_feature_columns` or `dataset.causal_feature_prefixes`.
- Extended method sweeps so a specific benchmark config can be swept with
  `--config`, and incompatible counterfactual methods can be skipped when a
  real feature table has no known causal/nuisance feature mask.
- Interpretation:
  the next SOTA-relevant experiment is now operationally clear. Once local
  Waterbirds features exist, run the benchmark sweep with all applicable
  methods, then only enable counterfactual methods if the feature schema
  provides a defensible causal feature mask.

## 2026-04-22: Direct Waterbirds Comparison Reporting

- Upgraded `scripts/report_benchmark_alignment.py` from a coarse alignment
  check into a direct comparison artifact.

## 2026-04-26: Gate Controls And Mechanism Reality Check

- Added and validated several grouped-gate mechanism variants on the real
  Waterbirds feature benchmark path:
  - score-conditioned gate offsets
  - contextual grouped gates
  - representation-conditioned grouped gates
  - disagreement-weighted counterfactual training
  - instability-replay counterfactual training with per-example EMA tracking
  - two-stage instability-JTT upweighting based on persistent counterfactual
    instability rather than ERM mistakes
- All of the above passed focused tests and full regression tests, but none beat
  the existing grouped discovery compact baseline.
- Latest compact top-128 grouped results:
  - grouped discovery: test WGA about 0.575
  - grouped scored: about 0.550
  - grouped conditioned: about 0.569
  - grouped contextual: about 0.570
  - grouped representation-conditioned: about 0.572
  - grouped disagreement-weighted: about 0.553
  - grouped instability-replay: about 0.544
  - grouped instability-JTT: about 0.662
- Full-budget promoted instability-JTT result:
  - grouped instability-JTT top-128 reached test WGA about 0.785 and val WGA
    about 0.797.
  - This beats grouped discovery top-128 at about 0.752 and grouped random
    top-128 at about 0.766.
  - It still trails the earlier fixed-gated discovery top-128 anchor at about
    0.790 by a small margin.
- Fixed instability-JTT hybrid check:
  - pairing instability-JTT training with the fixed discovery-gated consumer
    looked strong on the compact run at about 0.706 test WGA.
  - the promoted full top-128 hybrid only reached about 0.713 test WGA, well
    below fixed discovery top-128 and below grouped instability-JTT.
  - Interpretation: the compact win did not survive promotion, so the fixed
    consumer is not a free upgrade for the instability-based selection signal.
- Grouped instability-JTT compact sweep:
  - added `scripts/run_instability_jtt_sweep.py` to sweep stage-1 epochs,
    unstable-example fraction, and upweight under the compact budget.
  - best compact setting in the first sweep was stage1 epochs 20, top fraction
    0.15, upweight 3.0, with compact test WGA about 0.701.
  - promoted full run for that sweep winner only reached about 0.673 test WGA,
    so the compact winner did not survive promotion.
- Added `scripts/report_compact_promotion_alignment.py` and the generated
  `outputs/runs/waterbirds-compact-promotion-alignment.csv` artifact so compact
  screening versus full-run outcomes are explicit rather than anecdotal.
- Tightened compact promotion policy for follow-up sweeps:
  - compact candidates should only be promoted when both compact test WGA and
    compact val WGA clear thresholds and their gap stays small.
  - `scripts/run_instability_jtt_sweep.py` now records `promotion_score` and
    `eligible_for_promotion` directly in sweep outputs.
- Stability-aware stage-1 follow-up:
  - added a variance-penalized instability score mode,
    `counterfactual_instability_score_mode: mean_minus_std`, to reduce noisy
    stage-1 selections.
  - added grouped config
    `waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_stable_instability_jtt_gate_nuisance0p9.yaml`
    plus runner hooks for compact/full evaluation.
  - first compact sweep over the stable score mode produced no promotable
    candidates; the best compact promotion score was about 0.631, below the
    existing grouped instability-JTT compact baseline.
- Loss-aware stage-1 follow-up:
  - added `counterfactual_instability_score_mode: loss_weighted_mean` so the
    stage-1 selector can prioritize examples that are both counterfactually
    unstable and hard for the stage-1 model on the clean input.
  - added grouped config
    `waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_loss_weighted_instability_jtt_gate_nuisance0p9.yaml`
    plus runner hooks for compact/full evaluation.
  - first compact sweep over the loss-weighted score mode still produced no
    promotable candidates; the best compact promotion score was about 0.639,
    which is better than the stability-penalized follow-up but still below the
    existing grouped instability-JTT compact winner and below the promotion
    threshold.
- Counterfactual excess-loss stage-1 follow-up:
  - added `counterfactual_instability_score_mode:
    counterfactual_loss_increase_mean` so the selector prioritizes examples
    whose label loss actually worsens under the nuisance swap, not just
    examples with prediction drift.
  - added grouped config
    `waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_loss_delta_instability_jtt_gate_nuisance0p9.yaml`
    plus runner hooks for compact/full evaluation.
  - first compact check for that variant reached about 0.667 test WGA and
    about 0.647 val WGA, for promotion score about 0.647 with zero eligible
    rows.
  - Interpretation: the excess-loss selector is mechanistically cleaner than
    raw disagreement, but it still does not clear the grouped instability-JTT
    compact bar or the stricter promotion rule.
- Group-weighted counterfactual excess-loss follow-up:
  - added `counterfactual_instability_score_mode:
    group_loss_weighted_counterfactual_loss_increase_mean` so the selector can
    favor examples that are both counterfactually fragile and drawn from groups
    with high factual stage-1 loss.
  - added grouped config
    `waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_group_loss_delta_instability_jtt_gate_nuisance0p9.yaml`
    plus runner hooks for compact/full evaluation.
  - first compact check for that variant reached about 0.673 test WGA and
    about 0.639 val WGA, for promotion score about 0.639 with zero eligible
    rows.
  - Interpretation: adding group failure weighting improves compact test WGA a
    little but hurts validation alignment and still fails promotion. The
    stage-1 selector branch now looks locally exhausted.
- Backfilled missing fixed-gated full-budget controls and added grouped
  full-budget heuristic/random controls.
- Full-budget control signal is now decisive:
  - fixed gated discovery top-128: test WGA about 0.790
  - fixed gated random top-512: about 0.773
  - grouped learned-gate discovery top-128: about 0.752
  - grouped learned-gate random top-128: about 0.766
- Interpretation:
  the current discovery mechanism is not yet proving causal signal extraction
  beyond strong matched random-mask controls in the fixed-gated family, but the
  new grouped instability-JTT variant is the first grouped mechanism step that
  clears the matched grouped random-mask control. Future grouped-gate work
  should use instability-JTT as the new local comparator, while fixed-gated
  discovery remains the stronger overall anchor. The next principled mechanism
  step should modify the stage-2 counterfactual-adversarial objective rather
  than keep extending the stage-1 selector family.

## 2026-04-23: Discovery Learner Cleanup And Failure Mode

- Split discovery supervision into three categories:
  `explicit_mask`, `derived_mask`, and `none`.
- Marked synthetic and tiny fixtures as `explicit_mask` supervision sources.
- Marked Waterbirds feature masks produced by
  `label_minus_env_correlation` as `derived_mask` so the discovery learner no
  longer treats the heuristic mask as ground truth.
- Updated the discovery trainer to:
  - train only on rows with `has_explicit_supervision == true`
  - allow dataset exclusion from supervision
  - combine pointwise BCE with a pairwise ranking loss inside each dataset
- Added a learned-versus-heuristic overlap report for Waterbirds masks.
- Rebuilt the discovery dataset, retrained the scorer while excluding
  `waterbirds_features`, rescored Waterbirds, and reran the discovery-mask
  Waterbirds benchmark.
- Explicit-supervision training summary:
  - rows: `28`
  - datasets: `synthetic_linear`, `synthetic_nonlinear`, `dsprites_tiny`,
    `causal3d_tiny`
  - pointwise loss: about `0.348`
  - ranking loss: about `0.012`
  - train accuracy: about `0.786`
- Waterbirds mask-overlap summary:
  - heuristic size: `483`
  - learned top-k size: `512`
  - overlap: `205`
  - precision vs heuristic: `0.400`
  - recall vs heuristic: `0.424`
  - Jaccard: `0.259`
- Initial empirical result at that time:
  the cleaned-up discovery learner produced test WGA `0.522` and accuracy
  `0.882` on `waterbirds_features_counterfactual_adversarial_discovery_mask`.
- Later correction:
  that comparison turned out to be confounded by a downstream training
  regression in `fit_counterfactual_adversarial()`. When `nuisance_steps == 0`,
  the nuisance head was not optimizer-stepped on the main adversarial loss, so
  the schedule baseline and all discovery-mask variants collapsed to the same
  bad point.
- Post-fix validation:
  after restoring nuisance-head optimizer steps on the joint loss, the
  heuristic scheduled adversarial benchmark reran to test WGA `0.687` and
  accuracy `0.909`, while the direct learned-mask config reran to the same
  `0.687` / `0.909` point.
- Updated interpretation:
  the earlier `0.522` discovery conclusion was invalid because it measured a
  broken downstream method rather than a learned-mask failure. The current
  learned mask is therefore unresolved on Waterbirds: it is not yet better than
  the repaired heuristic baseline, but the post-fix direct comparison no longer
  shows the large regression that originally motivated the failure claim.
- The report now emits per-method Waterbirds rows with our latest WGA and
  accuracy, literature ERM/JTT/GroupDRO/DFR reference values, and gap-to-
  reference columns including gap to the best tracked WGA.

- Added explicit benchmark states to the report output:
  `real_benchmark_ready`, `fixture_only`, `blocked_missing_local_data`, and
  `no_literature_reference`.
- Extended the benchmark state machine so a real benchmark can also be blocked
  on missing provenance. Waterbirds comparisons now require documented feature
  extractor, feature source, and split definition before they are treated as
  literature-comparable.
- Regenerated the markdown research report so it now includes:
  - a blocked real benchmark section,
  - a direct comparison section for any real comparable runs,
  - and a dedicated non-comparable development section for Waterbirds-style
    fixture runs with per-method deltas to published references.
- Current Waterbirds status after this change:
  fixture methods can now be inspected against the literature gap structure in
  one place, while the real benchmark config is reported as blocked because the
  local feature table is still missing.

## 2026-04-23: Counterfactual-Adversarial Regression Repair

- Root-cause debugging of the regressed Waterbirds schedule rerun showed that
  the controlling issue was in `fit_counterfactual_adversarial()`, not in the
  discovery scorer or mask-construction path.
- The bug: when `nuisance_steps == 0`, the nuisance head accumulated gradients
  from the joint adversarial loss but never received an optimizer step, leaving
  the adversary effectively frozen.
- Fix applied:
  zero the nuisance optimizer before the joint backward pass, clip nuisance-head
  gradients alongside model gradients, and always call `nuisance_opt.step()` on
  the main adversarial update.
- Focused validation:
  - pre-fix schedule rerun: test WGA `0.522`, accuracy `0.882`
  - post-fix schedule rerun: test WGA `0.687`, accuracy `0.909`
  - historical best stored schedule run still remains higher at test WGA
    `0.782`, accuracy `0.917`
- Direct discovery comparison after the fix:
  `waterbirds_features_counterfactual_adversarial_discovery_mask` reran to the
  same test WGA `0.687` and accuracy `0.909` as the repaired heuristic schedule
  config.
- Interpretation:
  the main discovery result changed from "learned mask is much worse than the
  heuristic" to "current direct learned-mask config is performance-matched with
  the repaired heuristic baseline, while stronger-head schedule variants still
  define the higher local ceiling."

## 2026-04-23: Discovery Implementation Kickoff

- Added a first discovery-oriented implementation slice instead of jumping
  directly to a new downstream learner.
- New module: `causality_experiments.discovery.build_feature_clue_rows()` emits
  per-feature clue rows from an existing dataset bundle.
- Current clue rows include dataset/split metadata, per-feature label and
  environment correlation strengths, correlation margin, simple feature summary
  statistics, and any available ground-truth or proxy structural supervision
  already exposed by the bundle such as `causal_mask` or `cause_position`.
- Added `scripts/build_discovery_dataset.py` to turn a config into a persisted
  CSV artifact for the first discovery-training examples.
- Generated the first real benchmark artifact:
  `outputs/runs/waterbirds-feature-clues.csv`.
- Interpretation:
  this is the first concrete step toward learning causal discovery from clue
  generators already present in the repo. It does not yet learn a discovery
  model, but it creates the reusable supervision substrate needed for that next
  step.

## 2026-04-30: Multimodal Clue-Fusion Bridge

- Added the first reproducible clue-fusion bridge:
  feature-card generation, deterministic language clues, source score CSVs,
  source-ablation reporting, and a Waterbirds clue-fusion sweep runner.
- The bridge now handles opaque `feature_N` dimensions by using top/bottom
  activation alignment. Label-aligned features receive causal-language weak
  evidence; environment-aligned features receive spurious weak evidence. This
  avoids relying on semantic feature names that official Waterbirds feature
  tables do not have.
- On `data/waterbirds/features_official_erm_official_repro.csv`, source
  ablations showed the new language source is no longer neutral: language-only
  top-64 had mean language causal score `1.000`, mean language confidence about
  `0.814`, and only `26/64` overlap with the stats top-64. Fused top-64 had
  `32/64` overlap with stats and kept a stronger correlation margin than
  language-only.
- A compact downstream screen through plain `official_dfr_val_tr` produced
  identical metrics for stats, language, fused, heuristic, and random masks.
  Interpretation: plain official DFR does not consume the causal mask or soft
  feature scores, so it is not a valid clue-quality consumer.
- The same compact screen through `official_causal_shrink_dfr_val_tr` with
  `official_causal_shrink_prior: soft_scores` did consume the scores. With two
  DFR retrains, fused top-64/top-128 reached about `0.9282` test WGA versus
  about `0.9277` for stats/heuristic and `0.9273` for random.
- A higher-retrain fused-only check using the default 20 retrains tied across
  top-64, top-128, top-256, and top-512 at about `0.9315` test WGA and `0.9592`
  validation WGA. This is better than the compact random control, but still
  below the active `official_dfr_val_tr_retrains50` comparator at about
  `0.9330`.
- Interpretation: clue fusion now produces a measurable, auditable signal, but
  the current soft-shrink consumer is too conservative to surpass the official
  local comparator. The next mechanism iteration should use these clue scores
  inside a stronger objective or image/prototype bridge, not promote the current
  fused soft-shrink variant.

## 2026-04-30: Prototype Clues and Soft-Score Causal DFR

- Added image/prototype clue rows from feature-card activation evidence. The
  new source estimates whether a feature's top-activation prototype is more
  label-aligned or background-aligned, emits image label/background scores,
  group stability, prompt margin, and image confidence, and can be fused with
  language and statistical priors.
- Updated clue merging so multiple clue rows for the same feature compose
  instead of overwriting each other. Merged clue rows now preserve language,
  image, and bridge evidence together for v2 discovery vectors and score CSVs.
- Added `image` as a first-class source in
  `scripts/run_waterbirds_clue_fusion_sweep.py`. On official repro features,
  image top-64 had high mean image label score (`0.933`) and confidence
  (`0.716`) with only `9/64` overlap against stats, showing that it is a
  distinct clue source. Fused top-64 kept a stronger correlation margin than
  image-only while using both language and image evidence.
- Added `configs/benchmarks/waterbirds_features_official_causal_dfr_soft.yaml`,
  a stronger soft-score consumer that feeds clue-derived feature scores into
  `causal_dfr` via `causal_dfr_nuisance_prior: soft_scores`, rather than only
  scaling features for official causal-shrink DFR.
- Downstream screen on
  `data/waterbirds/features_official_erm_official_repro.csv`: stats, language,
  image, fused, and heuristic top-64/top-128 soft-score causal DFR all reached
  about `0.9401` test WGA and `0.9442` validation WGA; random controls were
  slightly lower at about `0.9397` test WGA. This clears the active local
  `official_dfr_val_tr_retrains50` comparator around `0.9330`.
- Interpretation: the stronger objective is the meaningful step-change; the
  image/prototype source is auditable and distinct, but the current downstream
  objective is not very sensitive to source ranking among plausible top-k
  priors. Promote the soft-score causal DFR path only after seed-stability
  checks, and continue improving ranking sensitivity.

## 2026-04-30: Discovery Score Soft-Selection Diagnostic

- Root cause for the tied top-k source screen: `discovery_score_top_k` changed
  the hard `causal_mask`, but soft-score consumers still received the full
  `causal_feature_scores` vector. That made top-k support less influential for
  `causal_dfr` with `causal_dfr_nuisance_prior: soft_scores`.
- Added `dataset.discovery_score_soft_selection: selected` as an opt-in data
  adapter mode. It keeps the hard top-k selection behavior and zeros soft
  scores outside the selected support, while preserving the old full-score
  behavior as the default.
- Added `--prune-soft-scores` to
  `scripts/run_waterbirds_clue_fusion_sweep.py` for ranking-sensitivity
  diagnostics. The default runner keeps full soft scores because that remains
  the stronger downstream path.
- Pruned diagnostic on official repro features through soft-score `causal_dfr`:
  heuristic top-64/top-128 stayed at about `0.9401` test WGA; image top-64 and
  random controls reached about `0.9397`; stats/language/fused mostly landed at
  about `0.9392`, and fused top-128 fell to about `0.9388`.
- Interpretation: pruning confirmed that source support can affect the soft
  objective, but the first pruned variant is not an improvement. Keep the
  full-score soft prior as the promotion candidate and use pruned mode only as
  a diagnostic for future ranking objectives.

## 2026-04-30: Clue Seed-Stability Gate

- Added `scripts/run_waterbirds_clue_seed_stability.py` to build clue-fusion
  score artifacts once, then run a paired baseline/candidate seed sweep over
  matched seeds. The runner writes raw rows, source ablations, a manifest, and
  a promotion-gate summary JSON.
- Ran a 3-seed check on official repro features for baseline
  `official_dfr_val_tr` versus full-score soft `causal_dfr` fused/heuristic/
  random top-64 candidates. The single-seed fused result at seed 101 still
  reached about `0.9401`, but seeds 102 and 103 dropped to about `0.9020` and
  `0.8754`. Mean fused test WGA was about `0.9058` versus the baseline at
  about `0.9315`, so this path is not promotable.
- Added `dfr_num_retrains` support to validation-split DFR/causal DFR. When
  set above 1, the method trains heads at seed offsets and averages the linear
  weights, mirroring the stabilizing idea used by official DFR retrains.
- Added `configs/benchmarks/waterbirds_features_official_causal_dfr_soft_ensemble.yaml`
  with `dfr_num_retrains: 3`. The 3-seed fused top-64 screen became much more
  stable, with mean test WGA about `0.9216`, std about `0.0057`, and minimum
  about `0.9143`, but it still trailed the official DFR baseline by about
  `0.0099` mean WGA.
- Tried an LBFGS head and a lighter nuisance weight as exploratory variants;
  both failed to improve the paired seed result. Interpretation: the next
  mechanism should not rely on the current validation-split causal DFR head as
  the promotion vehicle. Keep the seed-stability runner and retrain averaging
  as diagnostics, and move toward objectives that preserve the official DFR
  baseline while injecting clue priors more locally.

## 2026-05-01: Official Clue-Shrink DFR Screen

- Added `configs/benchmarks/waterbirds_features_official_clue_shrink_dfr_val_tr.yaml`,
  an official DFR head that consumes discovery soft scores through the existing
  causal-shrink feature-scale grid. The grid includes `1.0`, so the official
  tuner can fall back to the no-shrink baseline when the clue prior hurts.
- Ran `scripts/run_waterbirds_clue_seed_stability.py` on official repro
  features for fused top-64 with pruned soft scores over seeds 101/102/103.
  Baseline official DFR stayed at about `0.9315` test WGA. The clue-shrink
  candidate averaged about `0.9311` with std about `0.0019`, min about
  `0.9286`, and max about `0.9330`.
- Paired deltas were about `+0.0002`, `-0.0029`, and `+0.0016`, so the
  promotion gate failed. The selected shrink values were `0.7`, `0.7`, and
  `0.8`, confirming that the official tuner used the clue prior rather than
  always falling back to the baseline scale.
- Interpretation: official-compatible clue injection is the best current path.
  It preserves most of the official DFR strength and is far more seed-stable
  than validation-split causal DFR, but the clue support/scale grid needs more
  targeted tuning before promotion.

## 2026-05-01: Pivot To Upstream Conflict Sampling

- The official clue-shrink screen landed at noise-scale deltas, so the main
  research implementation shifted away from final-head/clue-grid tuning and
  toward representation/data interventions before official DFR.
- Added named ERM fine-tune sampling modes to
  `scripts/prepare_waterbirds_features.py`: `natural`, `group_balanced`,
  `conflict_upweight`, and `group_balanced_conflict_upweight`. The conflict
  modes upweight Waterbirds train examples where bird label and background do
  not match, which targets the minority/background-conflict groups before
  feature extraction rather than only in the final head.
- Extended `scripts/run_waterbirds_official_backbone_sweep.py` with
  `--sample-modes` and `--minority-weight`, plus feature/run tags such as
  `_conflictw3` and `_gbconflictw3`. The runner records the sampling mode in
  expected manifest settings so cached feature artifacts remain auditable.
- Verification: syntax checks passed, and focused feature-prep/backbone-search
  tests passed with `20 passed`. No full Waterbirds feature extraction was run
  yet; the next empirical step is a limited conflict-sampling screen before any
  full CPU run.

## 2026-05-01: Conflict Sampling Limit384 Screen

- Started the first upstream feature-generation experiment with
  `scripts/run_waterbirds_official_backbone_sweep.py` on a stratified limit384
  slice, seed 101, LR `0.001`, no environment adversary, and official DFR as
  the unchanged downstream evaluator.
- Conflict-only upweighting at minority weight `3.0` gave mixed results:
  - e3 feature artifact: base test WGA `0.75`, downstream official DFR test WGA
    `0.84375`. This improves over the old e3 no-conflict limit384 downstream
    result (`0.8125`) but only ties the old e5 no-conflict limit384 anchor.
  - e5 feature artifact: base test WGA `0.875`, downstream official DFR test
    WGA `0.75`, so extra epochs overfit or damage the downstream feature space.
- Lighter conflict-only upweighting at minority weight `1.5` for e3 was worse:
  base test WGA `0.84375`, downstream official DFR test WGA `0.8125`.
- Group-balanced conflict upweighting at weight `3.0` collapsed for e3 at the
  base-ERM stage: test WGA `0.0`, strong label-1 bias, and no downstream DFR
  run. The e5 grouped-conflict branch was stopped early after the e3 collapse
  and the conflict-only e5 regression made the setting unlikely to be worth the
  remaining CPU time.
- Interpretation: the conflict-sampling implementation works and can alter the
  representation, but these exact sampling weights are not a whole-point path.
  Avoid full runs for conflict-only weight `1.5`/`3.0` and grouped-conflict
  weight `3.0`; the next upstream experiment should change the training recipe
  more materially, for example a different backbone/source or a staged sampler
  rather than uniform conflict oversampling from epoch 1.

## 2026-05-01: Staged Conflict Sampling Limit384 Screen

- Added staged ERM fine-tune sampling to the Waterbirds feature-prep path. The
  trainer now accepts `erm_finetune_sample_warmup_epochs`, uses natural sampling
  before that epoch boundary, then switches to the resolved target sample mode;
  the setting is recorded in resolved manifests and in official backbone sweep
  tags such as `_conflictw3_samplewarm2`.
- Checked local Hugging Face vision caches for a fresh frozen-source screen.
  Only `facebook/dinov2-small` and `openai/clip-vit-base-patch32` were present
  among the searched families, and both were already negative in earlier
  limit384 diagnostics, so no new local frozen HF source was worth rerunning.
- Ran the first staged conflict screen with seed `101`, e5, LR `0.001`, no
  env-adv, conflict weight `3.0`, sample warmup `2`, CPU, and limit384. The
  training log confirms epochs 1-2 used `natural` sampling and epochs 3-5 used
  `conflict_upweight`.
- Result: manifest and base metrics cleared guardrails. Base ERM reached val
  WGA `0.6875` and test WGA `0.9375`, but downstream official DFR reached val
  WGA `0.875` and test WGA `0.8125`. The summary artifacts are
  `outputs/dfr_sweeps/official-backbone-staged-conflict-limit384.csv` and
  `outputs/dfr_sweeps/official-backbone-staged-conflict-limit384.json`.
- Interpretation: staging prevented the base-ERM collapse seen in the bluntest
  grouped-conflict setting and produced a much stronger base classifier, but it
  did not produce better DFR-ready features. This staged conflict recipe is not
  a full-run candidate; future Track B work needs a materially different
  representation mechanism, not more small conflict-sampling schedule tweaks.
- Verification: focused tests passed with `20 passed`, and the full regression
  suite passed with `131 passed`.

## 2026-05-01: Global Supervised Contrastive Feature Prep

- Added an opt-in global supervised-contrastive objective to the Waterbirds
  image feature-prep trainer. Positive pairs are same bird label across
  different backgrounds, while same-background/different-label examples are
  exposed as hard negatives in the denominator. The objective is disabled by
  default and is controlled by `erm_finetune_contrastive_weight`,
  `erm_finetune_contrastive_temperature`, and
  `erm_finetune_contrastive_hard_negative_weight`.
- Added `erm_finetune_seed` to the feature-prep path and wired the official
  backbone sweep seed into image fine-tuning. This fixed an experiment-control
  gap: previous Track B seed tags controlled DFR and filenames, but not the
  stochastic backbone fine-tune itself.
- Extended `scripts/run_waterbirds_official_backbone_sweep.py` with
  contrastive flags and `_supconw..._t...` tags, and added loss-component
  logging so future screens show CE and contrastive terms separately.
- Verification: focused feature-prep/backbone-search tests passed with
  `22 passed`, and the full regression suite passed with `133 passed`.
- First seeded limit384 diagnostics used e5, LR `0.001`, no env-adv, natural
  sampling, seed `101`, CPU, and a separate `data/waterbirds/seeded_screens`
  feature directory:
  - no-contrastive control: base test WGA `0.84375`, official DFR test WGA
    `0.71875`.
  - contrastive `w=0.05`, `t=0.2`: base test WGA `0.84375`, official DFR test
    WGA `0.71875`.
  - contrastive `w=0.2`, `t=0.15`: base test WGA `0.84375`, official DFR test
    WGA `0.71875`.
- Feature matrices were not identical: versus the seeded control, the weak
  contrastive run had feature-difference L2 about `1.76`, and the stronger run
  had L2 about `9.32`. So the objective is wired and moving the representation,
  but global label/background contrast alone did not improve the downstream DFR
  decision surface on the diagnostic slice.
- Interpretation: do not scale these global supervised-contrastive recipes to
  full Waterbirds runs. The next bird-background separation attempt should add
  localization/component structure, such as patch/object features or a cheap
  DINO/PCA component split, then reuse the official DFR and clue machinery on
  those decomposed representations.

## 2026-05-01: Decomposed DINO/Center-Background Features

- Added `feature_decomposition: center_background` to the Waterbirds feature
  export path. It runs the feature model over the full image, a center crop,
  and four corner crops, then exports concatenated full, center, averaged
  background, and center-minus-background features. The mode is opt-in and
  recorded in manifests and official backbone sweep tags as `_decompcenterbg`.
- Added a DINO/ViT-native frozen Hugging Face backbone mode,
  `hf_patch_components`, that pools full-image patch tokens into CLS,
  center-patch, corner-background, and center-minus-background components in one
  forward pass. This is much cheaper than six crop forwards and closer to the
  DINO-style patch decomposition idea.
- Verification: focused Waterbirds tests passed with `24 passed`, and the full
  regression suite passed with `135 passed`.
- Limit384 screens with seed `101` and official DFR scoring:
  - seeded e5 ResNet control from the prior round: official DFR test WGA
    `0.71875`.
  - seeded e5 ResNet with `center_background`: `0.78125` test WGA and
    `0.8984375` test accuracy.
  - frozen ResNet50 ImageNet-V2 with `center_background`: `0.75` test WGA.
  - frozen local DINOv2-small with crop `center_background`: `0.875` test WGA
    and `0.9375` test accuracy, including under the retrains50 DFR config.
  - frozen local DINOv2-small `hf_patch_components`: `0.875` test WGA and
    `0.9140625` test accuracy, also unchanged under retrains50.
- Feature dimensionality checks: ResNet crop decomposition exports `8192`
  features; DINO crop decomposition and DINO patch components each export
  `1536` features.
- No-limit seed101 screen for DINO patch components with
  `official_dfr_val_tr_retrains50` reached test WGA `0.9112149477005005` and
  accuracy `0.9368311762809753`, below the official local comparator around
  `0.9330`. The full crop-decomposed DINO export was stopped after the long
  run produced no feature artifact or prepare-stage output, so it needs a more
  efficient extraction path before it is a practical full-data screen.
- Interpretation: decomposition is the first branch in this round to move the
  limit diagnostic in the right direction, especially with DINO features, but
  the current patch-component full run is not a benchmark improvement. Future
  work should either make crop-DINO extraction efficient enough to finish or
  improve patch-token component selection beyond fixed center/corner pooling.

## 2026-05-01: Component-Aware Patch Intervention Infrastructure

- Started the component-causal implementation track. Waterbirds feature export
  now names decomposed columns by component group instead of only anonymous
  numeric indices. Examples include `feature_cls_*`, `feature_center_*`,
  `feature_foreground_*`, `feature_background_*`, and difference-component
  names. The prepared feature manifest now records `feature_columns` and
  `feature_components`, giving downstream discovery scores a stable map from
  feature name to component group.
- Added selector-style Hugging Face patch pooling modes on top of the existing
  fixed center/corner DINO path. `hf_patch_cls_components` pools top and bottom
  patch tokens by cosine similarity to the CLS token; `hf_patch_norm_components`
  pools top and bottom patch tokens by token norm. Both export CLS,
  foreground-like, background-like, and difference summaries through the same
  feature CSV path.
- Added `causality_experiments.patch_interventions`, a lightweight latent-token
  intervention module. It can select patches from CLS-similarity or token-norm
  scores, replace selected tokens with zero, image mean, donor tokens, or
  prototype centroids, rebuild hidden states, and summarize counterfactual
  target-logit deltas, prediction flip rates, correct-to-wrong rates, and
  group-conditioned effects.
- Interpretation: this does not yet claim a Waterbirds improvement. It creates
  the missing bridge between patch decomposition and causal discovery: patch
  components can now be named, probed, counterfactually edited, and converted
  into discovery-compatible evidence before downstream DFR or causal-shrink
  consumers use them.
- Verification: focused feature-prep and patch-intervention tests passed with
  `26 passed`; adjacent discovery/clue/sweep tests passed with `30 passed`;
  the full regression suite passed with `145 passed`.

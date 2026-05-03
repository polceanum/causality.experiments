# Research Log

Chronological notes about experiments, implementation choices, and empirical
signals. Keep this focused on what was tried and what was learned.

## 2026-05-02: Bridge Candidate Manifest and Support Audit

- Added `scripts/report_waterbirds_bridge_candidate.py`, which builds a
  checksum-backed promotion report from the paired bridge-fused random-control
  CSV/JSON artifacts, the score CSV directory, and the refreshed trace snapshot.
- Generated `outputs/dfr_sweeps/bridge-fused-candidate-report.md` and JSON from
  the existing `bridge_fused/w0.3/top512` five-seed run. The report recomputes
  the best-random gate from raw rows: candidate mean WGA `0.9367601395`, mean
  paired delta to official DFR `+0.0062305570`, mean delta to stats
  `+0.0031152844`, mean delta to best random by seed `+0.0021807075`, and
  non-negative best-random deltas on `5/5` seeds.
- Added `scripts/report_waterbirds_bridge_support.py` to compare score-selected
  top-k supports against a clue CSV. On current top-512 artifacts,
  `bridge_fused/w0.3` overlaps stats on `311/512` features (`0.4362` Jaccard),
  while deterministic random-score controls overlap only `124`-`145` features
  (`0.1378`-`0.1650` Jaccard).
- Support diagnostics suggest the bridge-fused candidate is not merely a random
  perturbation of stats: it preserves a large stats-compatible support while
  selecting far fewer env-dominant features (`5` with `env_corr >= label_corr`)
  than random controls (`91`-`93`). This points the next improvement branch
  toward support-composition filters and official-compatible score-to-scale
  transforms rather than tiny local blend/top-k sweeps.
- Focused tests passed for the new reporting scripts and existing bridge fused
  runner: `tests/test_report_waterbirds_bridge_candidate.py`,
  `tests/test_report_waterbirds_bridge_support.py`, and
  `tests/test_run_waterbirds_bridge_fused_sweep.py`.

## 2026-05-02: Bridge Support-Composition Iteration

- Extended `scripts/run_waterbirds_bridge_fused_sweep.py` with opt-in support
  variants for the bridge-fused score file: hard env filtering, margin gating,
  stats fill, soft env penalty, stats anchor, and monotone score-shape variants
  (`score_sqrt`, `score_square`). These variants keep the same official DFR
  comparator, stats control, random-score controls, and official causal-shrink
  consumer.
- Compact seeds `101`/`102`, five retrains, top-512: hard `env_filter` and
  `stats_fill` both improved over incumbent `bridge_fused/w0.3/top512` by about
  `+0.00044` mean WGA and cleared stats/random controls; `margin_gate`
  regressed.
- Full five-seed, 50-retrain gate for `env_filter` did not promote. The
  incumbent stayed better with mean WGA `0.9367601395`; `env_filter` reached
  `0.9361370683`, with mean delta to official DFR `+0.0056074858`, mean delta
  to stats `+0.0024922132`, but only `3/5` non-negative deltas against the
  best random control. Interpretation: hard shortcut filtering preserves some
  upside but over-prunes important support on seeds `103`/`104`.
- Softer follow-ups did not produce a larger compact margin. `soft_env_penalty`
  top-512 improved only about `+0.00022` over the compact incumbent;
  `stats_anchor` regressed. A top-k scan over `384/512/640/768` kept top-512 as
  the only useful region. Monotone score-shape variants `score_sqrt` and
  `score_square` tied the incumbent exactly in compact, suggesting the current
  official-shrink consumer is insensitive to those rank-preserving transforms.
- Stop rule: do not spend more full-budget compute on simple support filtering,
  stats anchoring, top-k widening/narrowing, or monotone score shaping around
  `w0.3/top512`. The next improvement branch should change bridge supervision
  or the official-compatible consumer objective more substantially.

## 2026-05-02: Offline RL Clue Policy Substrate

- Added `causality_experiments.rl_clue_policy`, which turns replayed latent
  clue packets, planner/test traces, and feature clue rows into explicit reward
  rows. Reward rows record schema version, dataset, reward scope,
  trainability, packet/action identity, immediate test reward, score-delta
  reward, control-pass reward, hypothesis reward, causal-target reward, and
  total reward.
- Added a hard training guardrail: rows marked as benchmark-final or Waterbirds
  test-scope rewards are not trainable and are rejected before fitting. The
  current reward table uses fixture/test-result artifacts only; final
  Waterbirds test WGA remains a reporting metric, not a trainable reward.
- Added `scripts/train_llm_clue_policy.py`, a leave-one-fixture-out evaluator
  for a small ridge value policy over packet/action features. It writes a
  reward CSV plus held-out recovery CSV/JSON and reports raw policy, stats,
  deterministic random, and normalized policy/stat fusions.
- Refreshed trace result, alpha `10`, top-k `1/2/4`: raw
  `offline_clue_policy` reached causal-target recovery `0.25/0.3125/0.28125`,
  below stats top-1 (`0.625/0.3125/0.28125`) but above random
  (`0.0/0.125/0.125`). Conservative `policy_stats_fused_w0.3` preserved stats
  top-1 at `0.625` and improved top-2/top-4 to `0.375`/`0.34375`.
- Interpretation: the RL reward substrate is now present, but naive offline
  value regression should not be promoted by itself. The useful next branch is
  pairwise/listwise or artifact-risk-aware training over the same reward table,
  with policy/stat fusion used as a conservative diagnostic candidate.

## 2026-05-02: Policy-Fused Downstream Screen

- Added explicit policy/stat fusion modes to `scripts/train_llm_clue_policy.py`.
  The key distinction is now visible in the evaluator: raw contextual-bandit
  value prediction remains weak at top-1, while conservative fusion can improve
  wider supports without sacrificing stats top-1.
- Added policy-derived Waterbirds score sources to
  `scripts/run_waterbirds_clue_fusion_sweep.py`: `policy`, `policy_fused`, and
  `policy_safe`. Also extended `scripts/run_waterbirds_bridge_fused_sweep.py`
  so `policy_fused` and `policy_safe` can be evaluated with the same paired
  official DFR, stats, and deterministic random-score controls used by the
  bridge-fused candidate.
- Fixture held-out result, alpha `10`: `policy_stats_safe_residual_w0.5`
  preserved stats top-1 causal recovery (`0.625`) and improved top-2/top-4 to
  `0.4375`/`0.34375`, compared with stats at `0.3125`/`0.28125`.
- Downstream compact result shifted the preference: `policy_safe` regressed,
  while `policy_fused/w0.5/top512` reached seed-101 compact test WGA
  `0.9348115325` versus stats `0.9334811568`. In the paired compact two-seed,
  five-retrain gate, `policy_fused/w0.5/top512` averaged `0.9346954823`, with
  mean deltas `+0.0019955635` to official DFR and `+0.0006651878` to stats, but
  only `1/2` non-negative deltas against the best deterministic random-score
  control.
- Interpretation: RL is not currently the best framing for the mechanism if it
  means standalone policy optimization. The useful substrate is logged reward
  provenance plus supervised rank/fusion learning. Policy-fused scores are a
  promising auxiliary signal, but the active `bridge_fused/w0.3/top512`
  candidate remains stronger and better controlled.

## 2026-05-02: Constrained Support Optimizer

- Added an opt-in constrained support selector to
  `scripts/run_waterbirds_bridge_fused_sweep.py`. The selector is top-k aware:
  it preserves a stats-ranked core, fills the remaining support with
  bridge-ranked features, caps env-dominant additions, and emits a score file
  whose selected top-k exactly matches the constructed support.
- Added four variants for compact screening: default `constrained_support`,
  `constrained_support_strict`, `constrained_support_loose`, and
  `constrained_support_bridge`. Focused tests cover paired-runner integration
  and the env-risk cap behavior.
- Compact paired two-seed, five-retrain screen at `w0.3/top512`: loose
  constrained support tied the incumbent `bridge_fused/w0.3/top512` exactly
  with mean WGA `0.9352525771`, mean delta `+0.0025526583` to official DFR,
  mean delta `+0.0012222826` to stats, and non-negative best-random deltas on
  `2/2` seeds. Strict and bridge-leaning variants regressed to mean WGA
  `0.9342520237` and were non-negative against best-random on only `1/2`
  seeds.
- Support overlap diagnostics explain the tie/regression: loose constrained
  support overlaps the incumbent on `438/512` features (`0.7474` Jaccard), while
  strict overlaps stats much more heavily (`422/512`, `0.7010` Jaccard) and
  loses bridge support. Interpretation: simple constrained support construction
  is safe but does not yet add information beyond the incumbent; the next use
  should be artifact-risk or active-boundary replacement, not more fixed core
  fractions.

## 2026-05-03: Artifact-Risk Boundary Head

- Added an opt-in artifact-risk scorer to
  `scripts/run_waterbirds_bridge_fused_sweep.py`. The scorer trains a small
  ridge head from replayed fixture traces, using shortcut geometry plus failed
  control, wrong-hypothesis, weak-delta, and low-test-value signals as the risk
  target. It writes risk-aware score files as either `artifact_risk` or
  `artifact_risk_boundary`, where the boundary variant only penalizes features
  around the active top-k cutoff.
- Added focused tests for sweep integration, boundary-only replacement, and a
  nonzero trained-risk baseline. The implementation keeps the ridge intercept
  active during standardization; an early version centered the intercept away
  and collapsed Waterbirds risk predictions to zero.
- Support diagnostics on the refreshed Waterbirds trace snapshot showed no
  change to the incumbent `bridge_fused/w0.3/top512` support. Across risk
  weights `0.25` through `2.0` and boundary windows from `0.15` through `0.5`,
  the top-512 overlap with the incumbent remained `512/512`; overlap with stats
  stayed `311/512` (`0.4362` Jaccard). The active support already has only five
  `env_corr >= label_corr` features, so this risk head has no useful boundary
  edits to make.
- Interpretation: artifact-risk scoring is now useful instrumentation and a
  guardrail, but it is not a margin widener for the current candidate. The next
  bridge-supervision attempt should use pairwise/listwise ranking or a stronger
  active-boundary test signal rather than spending downstream compute on these
  first risk variants.

## 2026-05-03: Pairwise Bridge Supervision

- Extended `scripts/train_llm_clue_bridge_ranker.py` with a pairwise ridge
  ranker. It compares trace rows within the replay corpus, trains on feature
  differences weighted by target deltas, and reports `pairwise_bridge_ranker`
  plus conservative `pairwise_stats_fused_w*` rows in the existing
  leave-one-fixture-out evaluator.
- Added reusable Waterbirds score-row support for `pairwise_bridge` and
  `pairwise_bridge_fused`, and let the paired official/stat/random sweep
  evaluate `pairwise_bridge_fused` with the same controls as the incumbent
  bridge-fused source.
- Refreshed fixture held-out evaluation: raw pairwise improved scalar bridge at
  top-1 (`0.375` versus `0.25`) but stayed below stats (`0.625`). Conservative
  pairwise/stat fusion preserved stats top-1; `pairwise_stats_fused_w0.3`
  improved top-2/top-4 recovery to `0.375`/`0.34375`, versus stats at
  `0.3125`/`0.28125`.
- Compact Waterbirds paired screens at top-512 did not promote. With two seeds,
  five retrains, and three deterministic random controls,
  `pairwise_bridge_fused_w0.3` averaged WGA `0.9334787428`, with mean deltas
  `+0.0007788241` to official DFR, `-0.0005515516` to stats, and
  `-0.0008869171` to the best random control. Nearby weights were also below
  stats/best-random: `w0.1` mean WGA `0.9333651066`, `w0.5` mean WGA
  `0.9326999187`.
- Interpretation: pairwise supervision is a better diagnostic framing than raw
  scalar bridge regression, but the first version does not improve the
  benchmark-facing support. Keep it as infrastructure and move next to stronger
  active-boundary evidence or richer listwise/query grouping.

## 2026-05-03: Active-Boundary Retesting

- Added an opt-in `active_boundary` support variant to
  `scripts/run_waterbirds_bridge_fused_sweep.py`. It identifies features around
  the active top-k cutoff, runs cheap `conditional_signal_check` tests on that
  local band, and reranks only the boundary with normalized evidence while
  leaving the stable core alone.
- Support diagnostics on refreshed Waterbirds `bridge_fused/w0.3/top512` showed
  that this variant actually moves the support: `51/512` features changed,
  overlap with the incumbent was `461/512` (`0.8188` Jaccard), and
  env-dominant selected features dropped from `5` to `2`. Overlap with stats was
  `308/512`, close to the incumbent's `311/512`.
- Compact paired two-seed, five-retrain Waterbirds screen did not promote.
  Incumbent compact mean WGA stayed `0.9352525771`. Active-boundary mean WGA was
  `0.9339246154`, with mean delta `+0.0012246966` to official DFR,
  `-0.0001056790` to stats, and `-0.0004410446` to the best deterministic
  random control. It was non-negative on only `1/2` seeds against stats and the
  best random control.
- Interpretation: active boundary tests are now strong enough to alter support,
  but this first conditional-signal-only evidence over-replaces useful bridge
  features. Future boundary testing needs a better target, likely model-effect
  tests or validation-loss-aware replacement scoring, before downstream budget.

## 2026-05-03: Model-Effect Active Boundary

- Added `active_boundary_model_effect` as a second top-k-specific support
  variant. Instead of relying on conditional feature correlations, it fits a
  lightweight balanced logistic probe on the current top-k support plus the
  local boundary band, then estimates boundary evidence from held-out train WGA
  damage, log-loss damage, and coefficient magnitude when each boundary feature
  is ablated.
- Support diagnostics showed a real intervention: refreshed Waterbirds
  `bridge_fused/w0.3/top512` overlap fell to `436/512` (`0.7415` Jaccard), so
  `76/512` selected features changed. This scorer increased env-dominant
  selected features from `5` to `8`, unlike the conditional-signal variant,
  which reduced them to `2`.
- Compact paired two-seed, five-retrain screen still did not promote. Incumbent
  compact mean WGA was `0.9352525771`; model-effect active-boundary mean WGA
  was `0.9349227548`. It improved seed 102 to `0.9376947284`, but seed 101
  regressed to `0.9321507812`, leaving mean deltas `+0.0022228360` to official
  DFR, `+0.0008924603` to stats, and `+0.0005570948` to the best random
  control, with only `1/2` non-negative stats and best-random seeds.
- Interpretation: model-effect ablation is a better aligned boundary target
  than conditional signal, but the current one-probe scorer is too noisy and
  can admit env-dominant replacements. The next boundary attempt should use
  paired replacement evaluation or ensemble the probe signal across splits
  before spending downstream budget.

## 2026-05-03: Split-Ensembled Model-Effect Boundary

- Added `active_boundary_model_effect_ensemble`, which averages the same
  boundary ablation evidence across five balanced train-split probe seeds
  before normalizing and reranking the boundary band. This keeps the scorer
  deterministic and cheap while testing whether the single-probe variant was
  failing mostly from split noise.
- Support diagnostics showed no top-512 stabilization benefit on refreshed
  Waterbirds `bridge_fused/w0.3/top512`: the ensemble selected the same top-512
  support as the single-probe model-effect scorer, with `436/512` overlap with
  the incumbent (`76/512` replacements) and env-dominant selected count `8`.
- Compact paired two-seed, five-retrain screen matched the single-probe result
  exactly: mean WGA `0.9349227548`, seed 101 WGA `0.9321507812`, seed 102 WGA
  `0.9376947284`, mean delta `+0.0008924603` to stats, mean delta
  `+0.0005570948` to the best random control, and only `1/2` non-negative
  stats/best-random seeds.
- Interpretation: split ensembling alone does not fix the model-effect boundary
  failure. The next viable boundary route needs paired replacement evaluation,
  an explicit env-risk guard, or a different downstream-aware target rather
  than averaging the same ablation signal.

## 2026-05-03: Env-Guarded Model-Effect Boundary

- Added `active_boundary_model_effect_env_guard`, which keeps the model-effect
  boundary ablation signal but subtracts shortcut risk from boundary evidence
  before normalization. The goal was to preserve the downstream-aware probe
  target while avoiding the env-dominant replacements admitted by the raw
  model-effect scorer.
- Support diagnostics looked much better than the raw model-effect variants:
  refreshed Waterbirds `bridge_fused/w0.3/top512` overlap was `507/512`
  (`0.9807` Jaccard), so only `5/512` features changed, and env-dominant
  selected features dropped from `5` to `4`. This is the first active-boundary
  scorer that is both nontrivial and restrained.
- Compact paired two-seed, five-retrain screen was encouraging. The env-guarded
  variant averaged WGA `0.9354743063`, slightly above incumbent compact
  `0.9352525771`, with mean deltas `+0.0027743876` to official DFR,
  `+0.0014440119` to stats, and `+0.0011086464` to the best random control. It
  was non-negative on `2/2` seeds for all gates.
- Full five-seed, 50-retrain promotion screen did not promote. Incumbent
  `bridge_fused/w0.3/top512` averaged WGA `0.9367601395`; env-guarded boundary
  averaged `0.9357854962`. The guarded variant stayed positive versus official
  DFR on all seeds but failed stats and best-random on seed 104, with minimum
  deltas `-0.0033155680` to stats and `-0.0048732162` to best random.
- Interpretation: an explicit env guard is the right direction for boundary
  edits, but the current local penalty mostly recovers the incumbent support
  and still harms one high-variance seed. Future work should evaluate paired
  replacements directly or learn an env guard from downstream paired outcomes,
  rather than relying on a fixed correlation penalty.

## 2026-05-03: Paired Replacement Boundary Probe

- Added `active_boundary_paired_replacement`, a stricter boundary scorer that
  compares outside-boundary challengers directly against the incumbent support
  feature they would replace. For each pair it retrains a lightweight balanced
  probe on the original top-k support and on the one-feature replacement
  support, then scores the replacement by held-out WGA delta, log-loss delta,
  and a small shortcut-risk penalty.
- The first loose accept rule exposed a useful failure mode: it accepted `9`
  top-512 replacements, but the deltas were microscopic log-loss changes with
  effectively zero WGA improvement. Compact downstream WGA regressed to
  `0.9345873892` versus incumbent `0.9352525771`, and seed 101 failed the stats
  gate.
- Tightening the accept threshold to require a meaningful pair delta rejects
  those noise-level swaps. On refreshed Waterbirds `bridge_fused/w0.3/top512`,
  the thresholded scorer accepts `0` replacements and therefore ties the compact
  incumbent exactly: mean WGA `0.9352525771`, mean deltas `+0.0025526583` to
  official DFR, `+0.0012222826` to stats, and `+0.0008869171` to the best
  deterministic random control, with all compact gates `2/2`.
- Interpretation: paired replacement is the right evaluation shape, but the
  current train-split logistic probe does not expose strong enough local
  replacement signal around the incumbent support. Keep the harness; the next
  useful version should learn pair acceptance from downstream paired outcomes or
  use a stronger validation-aware probe target.

## 2026-05-03: Replacement Calibration Dataset Pivot

- Added `scripts/build_waterbirds_replacement_calibration.py`, which joins a
  candidate score file, a reference score file, clue metadata, and paired
  downstream sweep rows into one calibration row. The row records support
  overlap/change counts, entered/left feature correlation summaries, optional
  paired-replacement accept/evict counts, and paired downstream deltas to
  official DFR, stats, and the best deterministic random control.
- Generated the first calibration table at
  `outputs/dfr_sweeps/waterbirds-replacement-calibration.csv` across compact
  support-filter, constrained-support, active-boundary, model-effect,
  env-guard, and paired-replacement variants. This makes the empirical pattern
  explicit: broad edits tend to regress, no-op or near-incumbent edits tie, and
  compact-only small-edit wins are not reliable promotion evidence because hard
  `env_filter`/`stats_fill` looked best in compact but failed the full gate.
- Interpretation: stop adding local boundary tricks as candidate mechanisms.
  The next principled mechanism should train or calibrate pair/support acceptors
  from this outcome table, with full-promotion labels where available and a
  conservative penalty for compact-only positives that have not survived the
  five-seed 50-retrain gate.

## 2026-05-03: Conservative Replacement Acceptor

- Added `scripts/train_waterbirds_replacement_acceptor.py`, a small ridge-based
  acceptor over replacement-calibration rows. Its target is the worst mean
  margin to stats and best-random controls. It reports leave-one-out
  predictions, subtracts residual uncertainty from fitted predictions, and only
  recommends rows that both clear gates and have enough paired outcomes.
- On the compact-only nine-row table, the raw conservative score still ranked
  small-change compact wins highest, which exposed the main failure mode: compact
  positives are not actionable labels. Tightened the acceptor with a default
  `min_outcome_count=5`, matching the promotion seed count.
- Built a promotion-aware table with full-gate `env_filter` and
  `active_boundary_model_effect_env_guard` rows plus compact-only diagnostics.
  The acceptor recommended no interventions. `env_filter` and env-guard kept
  positive mean worst-control margins, but both failed at least one full gate;
  compact-only rows were blocked by the outcome-count requirement.
- Interpretation: this is a principled negative result and a useful guardrail.
  Current evidence says do not alter `bridge_fused/w0.3/top512`; future support
  edits should first add full-gate outcome labels to the calibration table, then
  pass this conservative acceptor before any promotion claim.

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
- Added a lightweight latent-token intervention module for patch editing and
  counterfactual summaries. This branch was later pruned after the active
  patch-probe stack failed to transfer into seed-stable downstream WGA; the
  component-aware feature export remains, but the intervention module no longer
  exists in active code.
- Interpretation: this did not claim a Waterbirds improvement. The durable
  piece is the named component export; the later patch-editing bridge was
  pruned after failing to produce seed-stable downstream gains.
- Verification at the time: focused feature-prep and patch-intervention tests
  passed with `26 passed`; adjacent discovery/clue/sweep tests passed with
  `30 passed`; the full regression suite passed with `145 passed`.

## 2026-05-01: Selector Patch Pooling Screens

- Committed and pushed the component-causal infrastructure as `8e14e9d`
  (`Add Waterbirds component patch interventions`) after the full regression
  suite passed with `145 passed`.
- Ran frozen local DINOv2-small selector pooling on the stratified limit384
  screen with seed `101`, CPU, `official_dfr_val_tr_retrains50`, and feature
  artifacts under `data/waterbirds/component_screens`:
  - `hf_patch_cls_components`: official DFR test WGA `0.875`, test accuracy
    `0.9296875`.
  - `hf_patch_norm_components`: official DFR test WGA `0.84375`, test accuracy
    `0.9296875`.
- Interpretation: CLS-similarity selection ties the earlier fixed DINO
  center/corner patch diagnostic but does not improve it. Token-norm selection
  is worse. Neither selector warrants a full no-limit run as-is.
- Ran a compact clue/soft-shrink pass on the better CLS-similarity selector
  feature table. Stats, fused, heuristic, and most random candidates at top-k
  `64`, `128`, and `256` stayed at test WGA `0.875`; random top-128 dropped to
  `0.8125`. Fused top-64 selected only `47/64` of the stats top-64 features
  (Jaccard about `0.58`), so the clue bridge is changing the support, but the
  downstream official causal-shrink search selected shrink scale `1.0` for
  nearly every candidate.
- Ran the pruned-soft-score variant for stats and fused top-k `64`, `128`, and
  `256`. Fused stayed at `0.875`; stats top-256 dropped to `0.84375`.
- Interpretation: current selector pooling plus current soft-shrink is not the
  1-2% mechanism. The direct latent patch-intervention follow-up was tried and
  later pruned; see the failed-attempts log before reviving this path.

## 2026-05-01: Patch Probe Stack Pruned

- The active latent patch-probe stack was removed from active code after it
  failed to produce a seed-stable downstream improvement. The preserved lesson
  is now in `docs/failed-attempts.md`: learned and multi-hypothesis patch masks
  improved frozen-head counterfactual logit-drop diagnostics, but scalar scores,
  edited feature tables, priors, and direct effect objectives did not beat
  matched controls in a promotable way.

## 2026-05-01: LLM-Guided Clue-Probe Bridge Started

- Added the first offline implementation slice for bridging latent clue probes
  with an LLM-style idea generator. `latent_clue_packets` converts feature-card
  and discovery evidence into stable replayable candidate packets;
  `llm_clue_planner` constrains the planner to JSON hypotheses and a fixed
  test-action catalog; `llm_clue_bridge` turns packet/plan/result traces into
  trainable targets for hypothesis labels, test value, and score deltas.
- Added `scripts/run_llm_counterfactual_clue_probe.py` as a fixture-safe mock
  backend runner. It writes feature cards, latent clue packets, hypotheses,
  test specs, untested clue rows, training traces, and a manifest without
  calling hosted models or reviving the pruned patch-probe machinery.
- Interpretation: this is infrastructure for learning which proposed probes are
  worth running, not yet a Waterbirds improvement. The planned next layers are
  deterministic feature-level counterfactual test execution, replay/local LLM
  backends, and held-out bridge-ranker evaluation.

## 2026-05-01: Deterministic LLM-Clue Test Execution

- Added `counterfactual_clue_tests`, the first deterministic executor for the
  LLM-guided clue bridge. It runs feature mean/zero/shrink edits, same-label
  different-environment donor swaps, different-label same-environment donor
  swaps, and conditional signal checks, with deterministic random-feature
  controls and measured model/logit effects when a fitted model is available.
- Updated the LLM clue-probe runner so mock-planned test specs are executed by
  default against the configured method. The runner now writes `test_results.csv`
  and tested clue rows, and bridge traces receive nonzero `test_value` and
  `score_delta` labels when effects beat controls.
- Interpretation: the bridge can now start collecting empirical supervision for
  which LLM-proposed probe actions are useful. This remains fixture-safe
  infrastructure; Waterbirds use should begin with diagnostics and controls,
  not downstream promotion.

## 2026-05-01: LLM-Tested Clues vs Fixture Baselines

- Ran the mock LLM-tested clue probe across all eight lightweight fixture
  configs with `--card-top-k 8 --max-packets 8`, writing artifacts under
  `outputs/dfr_sweeps/llm_clue_fixture_experiments/`.
- Compared top-k known causal-target recovery for `llm_tested`, stats, and
  deterministic random score sources. Aggregate top-1 mean causal target was
  `0.625` for `llm_tested`, `1.000` for stats, and `0.000` for random. At
  top-2 it was `0.3125` for `llm_tested`, `0.6875` for stats, and `0.125` for
  random.
- Interpretation: the tested LLM loop is already above random but still below
  the simple stats baseline. It succeeds on the simple tabular and
  Waterbirds-style fixtures, but misses several factor/sequence fixtures by
  chasing high observed effect/correlation features instead of the known latent
  cause. Those misses are useful training examples for the bridge ranker: it
  needs to learn when a tested feature effect is shortcut/artifact evidence
  rather than causal support.

## 2026-05-02: Waterbirds DFR Config Health Check

- Audited the active Waterbirds DFR configs after the handoff cleanup. The
  legacy `waterbirds_features_dfr` config still loads and runs on
  `data/waterbirds/features.csv`, reaching test WGA `0.8971962332725525` and
  test accuracy `0.9193993806838989` on the local rerun. This is healthy as a
  historical anchor but not competitive with the official comparator.
- Fixed stale official config paths: the `waterbirds_features_official_*`
  benchmark configs now point at the actual local feature table
  `data/waterbirds/features_official_erm_official_repro.csv` rather than the
  missing `data/waterbirds/features_official_erm.csv`.
- Reran the official DFR configs after the path fix. The 20-retrain
  `official_dfr_val_tr` config reached test WGA `0.9314641952514648` and test
  accuracy `0.945978581905365`; the stronger 50-retrain comparator reached
  test WGA `0.9330217838287354` and test accuracy `0.9466689825057983`.
- Regenerated the benchmark-alignment report after filtering temporary
  `tmp_*` recheck configs out of benchmark-ready rows. The strongest real
  comparable Waterbirds row is now the official 50-retrain DFR comparator at
  formatted WGA `0.933`, versus the currently recorded literature SOTA/reference
  WGA `0.929`, for a formatted delta of `+0.004`.
- Interpretation: the Waterbirds DFR stack is working, but the benchmark-facing
  path is the official 50-retrain comparator. The open legacy DFR config should
  not be used as the promotion bar. This validates the official reproduction
  against the recorded reference, but it is a comparator/baseline result, not a
  new proposed-method win.

## 2026-05-02: Official Shrink Target Iteration

- Added `scripts/run_waterbirds_official_shrink_sweep.py`, a paired sweep
  runner for official-protocol causal-shrink DFR. It uses the same locked
  feature table and official DFR retraining path as the comparator, writes
  baseline/candidate paired rows to CSV, emits a JSON promotion summary, and
  records selected C/shrink model details when available.
- Ran an initial seed-101 target screen against
  `official_dfr_val_tr_retrains50` with near-identity shrink grid
  `1.0,0.99,0.975,0.95` over top-k/margin/prior variants. The screen was
  stopped after the first several rows because it already showed no upside:
  top-64 mask and soft-score variants tied the locked comparator at test WGA
  `0.9330217838287354`, while top-128 mask fell to `0.9314641952514648`.
- A tiny real-data smoke using 5-retrain shrink verified the runner and model
  detail plumbing: top-64/margin0/mask with grid `1.0,0.95` selected shrink
  `0.95` and reached test WGA `0.9321507811546326` versus the weaker 20-retrain
  baseline at `0.9314641952514648`. This is a useful smoke, but it is below the
  correct 50-retrain target and is not promotable.
- Checked whether the official comparator itself had an easy C-grid improvement
  by running a temporary expanded grid. It selected C `0.02` and dropped to test
  WGA `0.914330244064331`, so the current locked comparator grid is not the
  obvious bottleneck.
- Interpretation: near-identity official shrink is not the path to beat the
  `0.933` target. The next attempt should not keep tweaking final-head shrink;
  it should change the upstream representation signal or train the clue bridge
  so that score/mask support is better than the current stats-like priors.

## 2026-05-02: Held-Out LLM Clue Bridge Ranker

- Added `scripts/train_llm_clue_bridge_ranker.py`, a local ridge-ranker
  evaluator for the LLM clue bridge. It trains on packet/plan/result traces
  from all but one fixture, scores the held-out fixture's latent clue packets,
  and compares bridge, stats-margin, and deterministic-random rankings on the
  same candidate packet set.
- The first leave-one-fixture-out run used existing artifacts under
  `outputs/dfr_sweeps/llm_clue_fixture_experiments/` and `alpha=10.0`. Mean
  known causal-target recovery across eight held-out fixtures was:
  - top-1: bridge `0.625`, stats-margin `0.500`, random `0.000`.
  - top-2: bridge `0.4375`, stats-margin `0.3125`, random `0.125`.
  - top-4: bridge `0.34375`, stats-margin `0.250`, random `0.125`.
- Per-fixture top-1 signal is encouraging but uneven. The bridge matched stats
  on the simple tabular and Waterbirds fixtures, missed Dsprites and text-toy,
  but recovered the known causal feature on shapes-spurious and fewshot-NER
  where stats-margin selected non-causal shortcut features.
- Interpretation: this is the first held-out evidence that the trainable bridge
  can beat stats on candidate packet ranking. It is still fixture-level and
  small-data evidence, not a Waterbirds downstream claim. The next bridge step
  should expand packet/test coverage and then test whether bridge-ranked support
  improves official Waterbirds clue/shrink consumers against the locked
  `0.933` comparator.

## 2026-05-02: Bridge-Ranked Waterbirds Downstream Consumer

- Added `bridge` and `bridge_fused` as opt-in clue-fusion sources. `bridge`
  scores target latent clue packets with a fixture-trained ranker, while
  `bridge_fused` conservatively mixes the normalized bridge score with the
  existing stats score. Waterbirds fixture traces are excluded from bridge
  training by default when producing downstream Waterbirds scores.
- Compact 5-retrain screen against the official causal-shrink consumer showed
  that pure bridge scores were not enough: top-512 bridge reached test WGA
  `0.9330376983`, while stats top-512 reached `0.9334811568`. The source
  ablation showed pure bridge was selecting high label-correlation but also high
  environment-correlation features, especially at small top-k.
- The conservative `bridge_fused` source improved small-support downstream rows
  in the compact screen: top-64 `0.9312638640` versus stats `0.9158878326`, and
  top-128 `0.9325942397` versus stats `0.9312638640`. Top-512 remained slightly
  behind stats in the compact screen.
- Full 50-retrain check for `bridge_fused` top-512 produced test WGA
  `0.9345794320`. A fresh rerun of the locked official DFR comparator remained
  `0.9330217838`, and the matching 50-retrain stats top-512 control was
  `0.9314641953`.
- Interpretation: bridge-ranked support now improves a real Waterbirds
  downstream consumer by about `+0.00156` WGA over the locked official DFR
  comparator on seed 101. This is the first downstream win, but it is below the
  existing promotion-gate margin and must be paired across more seeds before it
  is treated as a benchmark claim.

## 2026-05-02: Paired Bridge-Fused Screening

- Added `scripts/run_waterbirds_bridge_fused_sweep.py`, a paired sweep runner
  that writes bridge-fused and stats score files, runs the locked official DFR
  baseline by seed, and reports candidate deltas against both official DFR and
  stats controls. Rows are flushed as each run finishes so long screens remain
  inspectable.
- Compact seed-101 screen around the prior winner confirmed that top-512 with
  bridge weight `0.2` is the only useful local blend. At five retrains,
  `w0.2/top512` reached `0.9330376983`; weights `0.15` and `0.25` collapsed to
  the compact baseline at `0.9308204055`.
- Two-seed 50-retrain paired screen for `w0.2/top512`:
  - seed 101: official DFR `0.9330217838`, stats control `0.9314641953`,
    bridge-fused `0.9345794320`.
  - seed 102: official DFR `0.9330217838`, stats control `0.9330217838`,
    bridge-fused `0.9330217838`.
  - mean bridge-fused WGA `0.9338006079`, mean delta to official DFR
    `+0.0007788241`, mean delta to stats `+0.0015576184`.
- Interpretation: the downstream bridge-fused candidate is now paired and
  non-negative on both tested seeds, with a small positive mean. It is still
  below the current promotion gate; next improvement should target either a
  stronger bridge target or a better consumer than final-head shrink, not just
  small weight changes around `0.2`.

## 2026-05-02: Bridge Score Refinement Stop-Rule

- Added an opt-in `bridge_gated` score source. Unlike `bridge_fused`, which
  linearly blends stats and normalized bridge scores, `bridge_gated` uses bridge
  as a multiplicative boost on stats-supported features. This tests whether the
  bridge should preserve the anti-environment statistical margin more strictly.
- Compact seed-101 screen for `bridge_gated` showed no improvement over the
  existing bridge-fused winner. Its best compact rows were top-640 with weights
  `0.1` or `0.2`, both at WGA `0.9330376983`. A 50-retrain check for
  `bridge_gated/w0.1/top640` tied the locked official DFR comparator at
  `0.9330217838` and therefore was not promoted.
- A 50-retrain top-k refinement for the existing `bridge_fused/w0.2` source
  confirmed top-512 remains the only useful support size: top-448 and top-576
  tied the comparator at `0.9330217838`, while top-512 reproduced
  `0.9345794320`.
- A 50-retrain weight refinement around top-512 confirmed the same sharp
  optimum: weights `0.18` and `0.22` tied the comparator, while weight `0.2`
  reproduced `0.9345794320`.
- Interpretation: the current final-head shrink consumer has a narrow bridge
  sweet spot, but local score-source/top-k/weight tweaks do not widen the gap.
  The next serious attempt should improve the bridge target itself or feed the
  bridge into a different downstream consumer.

## 2026-05-02: Refreshed Bridge-Fused Candidate

- Added `scripts/run_waterbirds_bridge_causal_dfr_sweep.py`, a paired runner
  for testing bridge-fused score files as `causal_dfr` soft nuisance priors. It
  runs the locked official DFR baseline, matched stats controls, and bridge
  candidates while streaming paired deltas to CSV/JSON.
- The causal-DFR consumer was a controlled negative. On seed 101,
  top-512 soft-score `causal_dfr` looked strong (`0.9401` at nuisance weight
  `10`), but the three-seed check over seeds `101`-`103` collapsed to mean WGA
  about `0.9060`. A five-retrain causal-DFR ensemble at nuisance weight `30`
  remained below the comparator with mean WGA about `0.9195`.
- Tried adding activation-gap/alignment fields to the bridge ranker as stronger
  supervision. Leave-one-fixture-out quality regressed: the refreshed fixture
  corpus gave bridge top-1 causal-target recovery `0.25` versus stats `0.625`,
  even after ridge-alpha checks. The source feature expansion was backed out;
  the refreshed ignored trace corpus remains a distinct experimental corpus.
- Reran the official-shrink bridge-fused path with refreshed fixture traces.
  The refreshed corpus used `scripts/run_llm_clue_fixture_experiments.py` with
  the default `max_packets: 16` shape. The old `w0.2/top512` setting tied the
  locked comparator on seed 101, so it is no longer the active refreshed-trace
  optimum. A compact screen found `bridge_fused/w0.3/top512`, and the full
  50-retrain seed-101 check reached test WGA `0.9376947284`, delta
  `+0.0046729445` to official DFR and `+0.0062305331` to the stats top-512
  control.
- Full 50-retrain paired seeds `101`-`103` for `bridge_fused/w0.3/top512` were
  all positive: mean WGA `0.9371755123`, mean delta `+0.0041537285` to
  official DFR, and minimum paired delta `+0.0031152964`.
- Extending to seeds `104` and `105` kept the candidate non-negative against
  stats and positive against the seed-matched official baseline. Combined
  seeds `101`-`105`: candidate mean WGA `0.9367601395`, min WGA
  `0.9361370802`, mean delta `+0.0062305570` to official DFR, and mean delta
  `+0.0031152844` to stats.
- Interpretation: this is the first refreshed-trace, five-seed Waterbirds
  bridge result with a promotion-sized gap over the locked comparator. Before
  treating it as a benchmark headline, freeze or version the replay trace/score
  generation and add matched random/control score checks under the same paired
  runner.

## 2026-05-02: Matched Random Controls for Refreshed Bridge Candidate

- Extended `scripts/run_waterbirds_bridge_fused_sweep.py` with deterministic
  random discovery-score controls. Unlike the older hard random-mask path, these
  controls write random score CSVs and run through the same top-k,
  `discovery_score_soft_selection: selected`, and official causal-shrink DFR
  consumer as the bridge-fused score file.
- Snapshotted the refreshed fixture corpus at
  `outputs/dfr_sweeps/llm_clue_fixture_experiments_20260502_refreshed`. A
  seed-101 full 50-retrain sanity check using `--bridge-input-dir` pointed at
  this snapshot reproduced the active candidate WGA `0.9376947284`.
- Ran the full 50-retrain matched-control screen for seeds `101`-`105`,
  `bridge_fused/w0.3/top512`, and three deterministic random score controls.
  The bridge candidate stayed at mean WGA `0.9367601395`, mean delta
  `+0.0062305570` to official DFR, and mean delta `+0.0031152844` to stats.
- Random controls were weaker on average. The best random-control mean WGA was
  `0.9317757010`; the other two means were `0.9293299079` and `0.9280115604`.
  Candidate minus best-random-by-seed had mean `+0.0021807075`, minimum `0.0`,
  and `5/5` non-negative paired seeds.
- Interpretation: the refreshed bridge candidate now clears official DFR,
  stats, and matched random-score controls under the same downstream consumer.
  The remaining benchmark-facing gap is packaging provenance: attach the trace
  snapshot or generated score file to a durable manifest before writing a final
  claim table.

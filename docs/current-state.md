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
  - `dfr`
  - `causal_dfr`
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
- DFR is now the benchmark-facing Waterbirds anchor on the fixed ResNet feature
  table. Tuned single-seed results are about `0.897` test WGA for plain `dfr`
  and about `0.900` test WGA for `causal_dfr`, versus the literature DFR WGA
  reference of `0.929`.
- DFR validation metrics are protocol diagnostics, not unbiased holdout metrics,
  because `dfr` and `causal_dfr` train their final classifier on validation
  groups.
- Current DFR-native follow-ups are implemented behind opt-in config flags:
  deterministic loss-weighted group balancing, group-weight power, DFR-head
  counterfactual consistency, and soft-score `causal_dfr` nuisance priors.
- First Waterbirds probes did not produce a promotable DFR variant: loss-weighted
  DFR fell to about `0.875` test WGA, counterfactual consistency tied the tuned
  anchor, hard/soft causal-DFR regularization sweeps tied at best around
  `0.900`, and sampler group-weight powers away from `1.0` degraded WGA.
- Additional DFR gap-closing probes also failed to improve the anchor:
  representation-level DFR with an ERM MLP encoder fell to about `0.713` test
  WGA, LBFGS full-batch logistic DFR overfit validation and stayed near `0.80`
  test WGA, validation-threshold calibration had no principled test upside over
  `causal_dfr`, exact balanced validation subsampling underperformed replacement
  sampling, and train-feature standardization dropped DFR/causal-DFR to about
  `0.875`/`0.872` test WGA.
- The Waterbirds feature-prep path now supports protocol-aligned feature export:
  `scripts/prepare_waterbirds_features.py --erm-finetune-epochs <n>
  --erm-finetune-mode layer4 --features-csv data/waterbirds/features_erm_layer4_e<n>.csv
  --config <matching-config> --overwrite-features`. This trains a target ERM
  ResNet classifier on the Waterbirds train split before exporting penultimate
  features, which is closer to the original DFR setup than frozen ImageNet
  features.
- The first protocol-aligned Waterbirds ERM-layer4 feature ladder did not close
  the DFR gap. Using `scripts/run_waterbirds_erm_feature_dfr.py --epochs 1,3,5
  --mode layer4 --batch-size 32 --lr 0.0001 --weight-decay 0.0001`, the best
  result was epoch 3 `causal_dfr` at about `0.893` test WGA and `0.941` test
  accuracy. Epoch 1 reached about `0.874` test WGA for both DFR variants; epoch
  5 regressed to about `0.877`/`0.875` test WGA for `dfr`/`causal_dfr`.
  Validation WGA stayed high around `0.938`-`0.942`, reinforcing that DFR
  validation metrics are protocol diagnostics, not model-selection holdouts.
  The fixed-feature tuned anchors (`dfr` about `0.897`, `causal_dfr` about
  `0.900`) remain stronger locally.
- An SGD plus train-time augmentation probe over the same layer4 fine-tuning
  path also failed to close the DFR gap. Using
  `scripts/run_waterbirds_erm_feature_dfr.py --epochs 3 --mode layer4 --tag
  sgd_aug --optimizer sgd --augment --batch-size 32 --lr 0.001 --weight-decay
  0.0001`, both `dfr` and `causal_dfr` reached about `0.889` test WGA, below
  the fixed-feature tuned anchors and the earlier Adam/no-augmentation e3
  `causal_dfr` run. This suggests the next protocol probe should change the
  trainable backbone scope or training recipe rather than repeating layer4-only
  DFR heads.
- Full-backbone SGD plus train-time augmentation at learning rate `0.001` also
  failed to close the gap. Using
  `scripts/run_waterbirds_erm_feature_dfr.py --epochs 3 --mode all --tag
  sgd_aug_all_lr1e3 --optimizer sgd --augment --batch-size 32 --lr 0.001
  --weight-decay 0.0001`, both `dfr` and `causal_dfr` reached about `0.877`
  test WGA despite about `0.929` test accuracy. The high average accuracy with
  weaker WGA suggests this setting is too aggressive or insufficiently robust
  for minority groups.
- Full-backbone SGD plus train-time augmentation at learning rate `0.0001` was
  also negative. Using `scripts/run_waterbirds_erm_feature_dfr.py --epochs 3
  --mode all --tag sgd_aug_all_lr1e4 --optimizer sgd --augment --batch-size 32
  --lr 0.0001 --weight-decay 0.0001`, both `dfr` and `causal_dfr` reached
  about `0.863` test WGA and about `0.904` test accuracy. This lower learning
  rate did not rescue full-backbone ERM features and performed below the
  `0.001` probe and the fixed-feature tuned anchors.
- Group-balanced full-backbone SGD plus train-time augmentation at learning
  rate `0.0001` partially repaired the full-backbone failure but still did not
  beat the fixed-feature anchor. Using
  `scripts/run_waterbirds_erm_feature_dfr.py --epochs 3 --mode all --tag
  gb_sgd_aug_all_lr1e4 --optimizer sgd --augment --balance-groups --batch-size
  32 --lr 0.0001 --weight-decay 0.0001`, `dfr` reached about `0.889` test WGA
  and `causal_dfr` reached about `0.893` test WGA. This validates group-aware
  backbone training as a useful protocol axis, but it remains below the tuned
  fixed-feature `causal_dfr` WGA around `0.900` and far below the literature DFR
  reference `0.929`.
- Group-balanced full-backbone SGD plus train-time augmentation at learning
  rate `0.001` produced the best local full-backbone single-seed result so far,
  but it still does not close the literature gap. Using
  `scripts/run_waterbirds_erm_feature_dfr.py --epochs 3 --mode all --tag
  gb_sgd_aug_all_lr1e3 --optimizer sgd --augment --balance-groups --batch-size
  32 --lr 0.001 --weight-decay 0.0001`, `dfr` reached about `0.903` test WGA
  and `causal_dfr` reached about `0.905` test WGA. This edges past the fixed
  feature `causal_dfr` anchor around `0.900`, but only by a small single-seed
  margin and it still trails the literature DFR reference `0.929` by about
  `0.024`. Treat it as a diagnostic lift, not a promotable closure of the gap.
- The repo now has a separate official-aligned comparator path for Waterbirds
  DFR reproduction: `configs/benchmarks/waterbirds_features_official_dfr_val_tr.yaml`
  plus `scripts/run_waterbirds_official_dfr.py`. This path is intended to match
  the published `DFRVal^Tr` protocol more closely than the local Adam-based DFR
  head, and should be the main benchmark-facing comparator until we know
  whether the remaining gap is mostly protocol or representation.
- The official-aligned Waterbirds comparator now works end to end on local
  data. Using the exported feature table
  `data/waterbirds/features_official_erm_official_repro.csv`, plain
  `official_dfr_val_tr` reached about `0.931` test WGA, slightly above the
  literature DFR reference `0.929`. This closes the earlier protocol gap: the
  benchmark problem is no longer “why is local DFR below reported DFR,” but
  “which mechanism can reliably beat the official local comparator.”
- The official comparator audit found one meaningful protocol update. Raising
  `official_dfr_num_retrains` from `20` to `50` increased paired test WGA on
  seeds `101`-`103` from about `0.9315` to about `0.9330`, with a consistent
  paired delta of about `+0.0016`; adding train examples to DFR hurt, averaging
  about `0.9232` test WGA. Treat `official_dfr_val_tr_retrains50` as the
  stronger local comparator for future Track A promotion decisions, not as a
  proposed-method win.
- Official-feature mechanism transfer results are now mixed in a useful way:
  single-seed `causal_dfr` reached about `0.939` test WGA on the official
  feature table, but a seed-matched comparison over seeds `101`-`105` averaged
  about `0.923` test WGA versus about `0.929` for plain `official_dfr_val_tr`.
  Current `causal_dfr` therefore has upside but is not yet a stable win over
  the official comparator.
- Track A now has a paired official-feature causal DFR runner,
  `scripts/run_waterbirds_official_causal_dfr_sweep.py`, that reports
  same-seed official-baseline deltas and streams rows to CSV as each fit
  completes. A seeds `101`-`103` smoke on the current causal DFR setting
  reproduced the instability: mean test WGA about `0.9252`, mean paired delta
  about `-0.0062`, and minimum paired delta about `-0.0140`. The interrupted
  broader grid showed the same seed101-win/seed102-loss pattern before it was
  stopped, so broad unpaired-looking causal DFR knob sweeps are not the next
  best use of compute.
- A new `official_causal_shrink_dfr_val_tr` method is implemented as a
  conservative official-protocol extension: it uses the same validation split,
  C-grid, balanced validation subsampling, and retrain averaging as
  `official_dfr_val_tr`, but can shrink nuisance dimensions after
  standardization before the L1 head fit. Shrink `1.0` exactly recovers the
  official DFR path, and run artifacts now persist `model_details` such as the
  selected C and shrink value. The first paired three-seed smoke is negative:
  on seeds `101`-`103`, causal-shrink DFR reached mean test WGA about
  `0.9299`, mean paired delta about `-0.0016`, and zero non-negative seed
  deltas versus the locked official comparator. It should not be promoted in
  its current grid.
- Softer causal-shrink grids do not yet clear the updated bar. A gentle hard
  mask grid (`1.0,0.95,0.9,0.85,0.75`) averaged about `0.9315` WGA against the
  20-retrain comparator, with paired deltas `+0.0016,0.0,-0.0016`; a soft-score
  prior averaged about `0.9309`. Repeating the gentle hard-mask grid with
  `official_dfr_num_retrains: 50` averaged about `0.9325` versus the stronger
  `0.9330` comparator, with paired deltas `-0.0016,0.0,0.0`. Causal shrink
  lowers nuisance-to-causal importance, but in these settings that diagnostic
  gain is not a benchmark improvement.
- Feature-swap `counterfactual_adversarial` remains non-competitive on the
  official feature table. Plain official-feature transfer collapsed to about
  `0.796` test WGA, a nuisance-zero ablation partially repaired it to about
  `0.883`, and enlarging the causal mask did not rescue the method. This
  strongly suggests the current random nuisance-swap construction is too
  off-manifold for deep official Waterbirds features.
- A new `official_representation_dfr` path is now implemented to test
  representation learners against the exact official DFR downstream protocol.
  The first swap-free adversarial representation probe underperformed:
  `adversarial_probe` plus official DFR reached about `0.913` test WGA with
  very poor selectivity (about `0.081`) and high nuisance-to-causal importance
  (about `0.886`). After wiring fixed causal input gating into
  `adversarial_probe`, a gated rerun improved diagnostics substantially
  (selectivity about `0.314`, nuisance-to-causal importance about `0.170`) but
  still fell to about `0.909` test WGA. The representation-level lesson so far
  is that reducing nuisance reliance alone is not enough; the next mechanism
  should preserve causal signal while suppressing nuisance, not just hard-mask
  it.
- Waterbirds benchmark status should now be read in three tiers:
  - official baseline:
    `official_dfr_val_tr_retrains50` on
    `features_official_erm_official_repro.csv` at about `0.933` test WGA is the
    stronger local comparator; the previous 20-retrain comparator at about
    `0.931` remains useful for historical continuity only.
  - promising but unstable:
    official-feature `causal_dfr` and any single-seed mechanism variant that
    beats the baseline but does not yet hold up under seed-matched averages.
  - promoted:
    only a candidate with seed-matched mean test WGA above the official
    baseline and acceptable variance should be treated as a real improvement.
- The repo now has three official Waterbirds search tracks:
  - `scripts/run_waterbirds_official_causal_dfr_sweep.py` for seed-matched
    official-feature causal DFR mask/nuisance sweeps with paired deltas.
  - `scripts/run_waterbirds_official_representation_sweep.py` for seed-matched
    official-feature mechanism sweeps with CSV/JSON summaries.
  - `scripts/run_waterbirds_official_backbone_sweep.py` for official-aligned
    backbone/feature-generation sweeps scored by plain official DFR.
- Track B now has a provenance guardrail. Waterbirds feature manifests record
  resolved ERM/backbone settings, and the official backbone sweep reports
  `manifest_settings_status` plus `base_metric_status`. Rows with missing
  settings or collapsed base ERM WGA are listed under `blocked_rows` and are
  skipped before downstream official DFR scoring.
- The existing broken Track B artifact
  `features_official_e50_lr0.001_envadv0_seed101.csv` is now formally blocked
  by this audit path: its cached manifest lacks resolved settings and its base
  ERM WGA is `0.0` on train, validation, and test. The audit artifact is
  `outputs/dfr_sweeps/official-backbone-audit-e50-envadv0-seed101-20260428.json`.
- A fresh rerun of that same no-env-adv Track B candidate reproduced the
  collapse with truthful settings: `manifest_settings_status=ok`, but
  `base_metric_status=blocked_base_erm:train/worst_group_accuracy,val/worst_group_accuracy,test/worst_group_accuracy`.
  The rerun artifact is
  `outputs/dfr_sweeps/official-backbone-rerun-e50-envadv0-seed101-20260428.json`.
  This means the blocker is now a live training/export issue, not merely stale
  provenance. The local `auto` device path also emitted repeated Metal/MPS
  command-buffer failures during the rerun, so the backbone sweep now defaults
  to CPU unless `--device auto` is requested explicitly.
- One concrete Track B training-path bug is fixed: official-style custom
  schedules now choose the official train-time augmentation based on
  `eval_transform_style="official"`, not on `erm_finetune_preset="official"`.
  This matters because Track B needs custom epochs/LR/env-adv settings without
  accidentally falling back to generic aggressive `RandomResizedCrop`.
  `tests/test_prepare_waterbirds_features.py` covers this path.
- Track B now has a cheap CPU diagnostic mode: `--limit <n>` on
  `scripts/run_waterbirds_official_backbone_sweep.py` uses a stratified
  split/group metadata slice and includes `_limit<n>` in feature/output tags so
  debug artifacts cannot be confused with full benchmark features. Base ERM
  manifests now also record per-group accuracies and predicted-label counts.
  On 2026-04-29, the corrected CPU path behaved sensibly on a 48-row slice:
  - e1 no-env-adv was undertrained and predicted mostly label 0
    (`14/16`-`15/16` predictions), giving group 3 WGA `0.0`.
  - e3 no-env-adv cleared the guardrail with base test WGA `0.75` and
    downstream official DFR test WGA `0.75`.
  Scaling the same corrected e3 recipe to limit96, limit192, and limit384 kept
  the guardrail green but exposed undertraining/label-bias behavior: downstream
  official DFR test WGA was `0.75`, `0.8125`, and `0.8125`, while the limit384
  base validation WGA fell to `0.1875` with heavy label-0 prediction bias.
  Increasing only epochs at limit384 was more informative: e5 improved base
  test/validation WGA to `0.84375`/`0.71875` and downstream official DFR test
  WGA to `0.84375`; e10 improved base test/validation WGA further to
  `0.9375`/`0.8125` but downstream official DFR fell back to `0.8125`. These
  diagnostics argue against a basic label/index/export bug and point to e5
  limit384 as the current Track B promotion candidate for a scheduled full CPU
  seed101 run. Limited artifacts remain diagnostics only and must not be used
  for benchmark claims.
- Full Track B CPU runs now emit JSON progress from the ERM fine-tuning loop
  and save a sidecar epoch checkpoint at
  `data/waterbirds/<features>.csv.training.pt` while training. A successful
  feature export removes the checkpoint after the manifest is written. This is
  an operational guardrail for long no-limit runs, not a change to the training
  objective or benchmark protocol.
- The first no-limit Track B CPU promotion candidate completed cleanly but did
  not beat the locked official comparator. Using e5, LR `0.001`, no env-adv,
  seed `101`, and CPU, the manifest cleared both guardrails
  (`manifest_settings_status=ok`, `base_metric_status=ok`) with base ERM test
  WGA `0.7757009267807007` and validation WGA `0.7142857313156128`. Plain
  downstream `official_dfr_val_tr` reached test WGA `0.914330244064331` and
  test accuracy `0.9295822978019714`, below the locked comparator test WGA
  `0.9314641952514648`. The artifact is
  `outputs/dfr_sweeps/official-backbone-cpu-rerun-e5-envadv0-seed101-20260429.json`.
  Do not promote this setting or run a seed sweep for it; the next Track B
  branch should change one mechanism axis at a time, starting with
  group-balanced backbone training.
- The first group-balanced Track B follow-up was negative on the same diagnostic
  slice that selected e5. The runner now includes `_gb` in group-balanced
  feature tags so balanced and non-balanced artifacts cannot collide. With e5,
  LR `0.001`, no env-adv, seed `101`, and `--limit 384 --balance-groups`, the
  artifact cleared guardrails but had base validation/test WGA `0.5`/`0.78125`
  and downstream official DFR test WGA `0.8125`, below the earlier no-balance
  e5 limit384 diagnostic `0.84375`. Do not launch a full group-balanced e5 run
  from this evidence; the next Track B mechanism-axis diagnostic should be a
  small env-adv probe.
- The first env-adv Track B follow-up was also negative on limit384. With e5,
  LR `0.001`, env-adv `0.05`, seed `101`, and no group balancing, the artifact
  cleared guardrails and improved base test WGA to `0.875`, but base validation
  WGA was only `0.53125` and downstream official DFR test WGA fell to `0.75`.
  This is the wrong tradeoff for the current promotion goal; do not launch a
  full env-adv `0.05` e5 run from this evidence.
- The Track B runner now supports frozen alternate torch-hub and Hugging Face
  vision backbones with explicit weights/eval-transform tags, while keeping ERM
  fine-tuning limited to the ResNet path. Available stronger-source diagnostics
  were negative on the limit384 slice when scored by the stronger
  `official_dfr_val_tr_retrains50` comparator: frozen ResNet50 ImageNet-V2
  features with weights transforms reached downstream WGA `0.84375`, frozen
  ConvNeXt-Tiny ImageNet-V1 features reached `0.8125`, frozen CLIP ViT-B/32
  features reached `0.625`, and frozen DINOv2-small features reached `0.84375`.
  Do not scale these frozen sources to full benchmark runs.
- A stricter training sanity check confirms the Track B ResNet path can train
  properly when the LR is less aggressive, but this still does not create a
  downstream win. On the limit384 diagnostic slice, e20/LR `0.0003` completed
  all 20 epochs with mean train loss falling to about `0.211`; the checkpointed
  base classifier reached train/test WGA `0.9375`/`0.9375` with balanced label
  predictions. Exporting penultimate features from that checkpoint and scoring
  with `official_dfr_val_tr_retrains50` reached only test WGA `0.8125` and test
  accuracy `0.8359375`. So the current failure is not simply that the model did
  not train; the trained representation is not helping the DFR head on the
  diagnostic slice. The full no-limit e20/LR `0.0003` CPU run was stopped after
  one completed epoch because it was too slow for an interactive local probe;
  use checkpointed/resumable scheduled runs for any future full-data training.
- The first new Track A configs are:
  - `waterbirds_features_official_adv_representation_dfr_score_gate`
  - `waterbirds_features_official_adv_representation_dfr_nuisance_regularized`
  These use the same official feature table and downstream DFR protocol as the
  baseline, but swap in soft score-guided gating or explicit nuisance
  regularization during representation learning.
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
2. Stabilize and improve the DFR anchor.
   - Use `scripts/run_dfr_sweep.py` for narrow, predeclared DFR sweeps and CSV
     outputs under `outputs/dfr_sweeps/`.
   - Treat tuned `dfr` and `causal_dfr` as the immediate comparators for any
     Waterbirds improvement claim.
   - Promote only seed-stable improvements over the `0.897`/`0.900` tuned
     anchors; single-run ties or tiny lifts should remain diagnostic.
   - Do not select by DFR validation WGA, since validation groups are used for
     retraining.
   - The current evidence points away from final-head tweaks on the fixed
     feature table, away from layer4-only ERM feature ladders, away from plain
     full-backbone SGD plus augmentation at LR `0.001` or `0.0001`, and now
     away from no-limit Track B e5/no-env-adv at LR `0.001`. That full CPU
     seed101 run cleared guardrails but reached only `0.914330244064331`
    downstream official DFR test WGA, below the locked official comparator
    `0.9314641952514648`. The first group-balanced and env-adv e5 limit384
    diagnostics were also negative (`0.8125` and `0.75` downstream WGA), so
    full runs for those exact settings are not justified. The remaining
    plausible gap is likely a different backbone-training recipe, a different
    validation-feature source, or a stronger backbone, not another small DFR
    objective variant or a seed sweep of the failed e5 variants.
   - Do not promote the ERM-layer4 e1/e3/e5, SGD-augmentation layer4 e3,
     full-backbone SGD-augmentation LR `0.001`/`0.0001`, or group-balanced
     full-backbone SGD-augmentation LR `0.0001` feature tables as improvements;
     they were diagnostic but remained below the fixed-feature tuned anchor.
3. Compose the strongest local mechanisms.
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
4. Develop factor/token-specific probe interventions.
   - Generic adversarial hiding is too blunt for sequence fixtures.
   - Use known factor/token metadata to design targeted intervention losses.
5. Use group-balanced ERM as a required comparator for any known-group result.
6. For every serious result, compare against literature reference/SOTA numbers
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

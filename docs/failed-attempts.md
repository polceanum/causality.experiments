# Failed Research Attempts Log

Record research attempts that were weak, misleading, or empirically failed.
Do not use this file for incidental tooling, git, environment, or debugging
issues unless they invalidate an experimental result.

## 2026-04-22

- **Over-regularized IRM as a general robust baseline**
  - Attempt: run IRM across all 8 fixture experiments with
    `penalty_weight=50.0`.
  - Result: many runs collapsed, with poor average accuracy and near-zero
    worst-group accuracy on several tasks.
  - Interpretation: the current IRM implementation is highly penalty-sensitive;
    a large invariant penalty can dominate the label objective instead of
    encouraging causal features.
  - Action: changed the sweep default to `penalty_weight=1.0` and longer
    training. Treat IRM as a tuned baseline, not a reliable default.

- **Plain MLP treatment for token/NER-style fixtures**
  - Attempt: represent token sequences as normalized float vectors and train the
    same MLP used for tabular fixtures.
  - Result: `07_text_toy` remains best under ERM with weak worst-group accuracy,
    and `08_fewshot_ner` only shows a small IRM improvement.
  - Interpretation: this setup under-models sequence structure and does not
    provide a meaningful test of causal language interventions.
  - Action: replace with an embedding-based sequence model and token-specific
    counterfactual augmentation before drawing research conclusions from these
    tasks.

- **Naive counterfactual augmentation on factor fixtures**
  - Attempt: preserve causal dimensions and randomly swap nuisance dimensions
    within a batch for all datasets with a causal mask.
  - Result: strong gains on clean spurious-feature tasks, but weak gains on
    `03_dsprites_3dshapes` and mixed or negative effects on `04_causal3dident`
    and `07_text_toy`.
  - Interpretation: mask-based nuisance swapping is too crude for structured
    factor or sequence settings. It works when the causal/nuisance split is
    simple and independent, but can generate invalid or uninformative
    counterfactuals elsewhere.
  - Action: add dataset-specific intervention policies rather than using one
    augmentation rule everywhere.

- **ATE proxy as a causal-effect metric**
  - Attempt: estimate an ATE-style diagnostic by forcing one causal feature from
    0 to 1 and measuring prediction change.
  - Result: useful as a smoke-test diagnostic, but values are hard to interpret
    across normalized token, factor, and tabular fixtures.
  - Interpretation: the metric is not yet a reliable causal-effect estimator.
  - Action: keep it labeled as a proxy; add task-specific effect metrics before
    reporting causal-effect claims.

- **Single-run sequence improvements as evidence**
  - Attempt: judge the sequence upgrade from one focused sweep.
  - Result: the single-run results looked encouraging, but three-seed sweeps
    showed high variance, especially for few-shot NER.
  - Interpretation: the sequence fixtures are still noisy and should not support
    claims from isolated runs.
  - Action: use seed-sweep mean/std reporting for sequence claims and add
    stronger sequence baselines before treating improvements as meaningful.

- **IRM as the primary known-group robustness baseline**
  - Attempt: compare new methods mainly against ERM and IRM.
  - Result: after adding group-balanced ERM and GroupDRO, IRM is clearly not the
    strongest known-group comparator on the current fixtures.
  - Interpretation: IRM is still useful as an invariance baseline, but it is too
    weak and penalty-sensitive to serve as the main robustness target.
  - Action: future progress claims should include group-balanced ERM and, where
    group labels are available, GroupDRO-style training.

- **JTT as a universal robustness fix**
  - Attempt: add a local JTT-style two-stage baseline and test it on
    Waterbirds-style and sequence fixtures.
  - Result: JTT is excellent on the Waterbirds-style fixture, but it loses to
    group-balanced ERM on text-toy mean WGA and underperforms on few-shot NER.
  - Interpretation: upweighting mistakes works well when early mistakes expose
    minority groups, but it can overcorrect or amplify noisy sequence errors.
  - Action: keep JTT as a required baseline for clean spurious-correlation
    settings, but do not treat it as evidence for causal detection by itself.

- **Linear probes as sufficient causal detection**
  - Attempt: add causal/nuisance linear probes on hidden representations and
    inspect selectivity.
  - Result: probes are diagnostic, but nuisance information remains decodable
    even in strong robust models. On text fixtures, nuisance is more decodable
    than the causal token across current methods.
  - Interpretation: decodability is not equivalent to model reliance or causal
    use.
  - Action: use probes as diagnostics and as ingredients for future
    intervention/regularization, not as standalone proof of causal detection.

- **Generic adversarial probe training on sequence fixtures**
  - Attempt: use gradient reversal to hide environment labels from pooled
    sequence representations.
  - Result: Waterbirds-style performance was strong, but text-toy WGA stayed at
    ERM level and nuisance remained more decodable than causal token identity.
  - Interpretation: sequence fixtures need token-aware interventions or
    regularizers; pooled representation adversarial training is too blunt.
  - Action: keep `adversarial_probe` as a promising image/tabular-style method,
    but develop factor/token-specific probe interventions next.

## 2026-04-30

- **Plain official DFR as a clue-fusion downstream consumer**
  - Attempt: evaluate stats, language, fused, heuristic, and random top-k clue
    masks through `official_dfr_val_tr` on official Waterbirds repro features.
  - Result: all candidates returned identical validation/test metrics.
  - Interpretation: plain official DFR does not consume `dataset.causal_mask`
    or `causal_feature_scores`, so this screen cannot measure clue quality.
  - Action: use `official_causal_shrink_dfr_val_tr` with soft-score priors, or
    another method that actually consumes clue scores, for downstream clue
    screens.

- **Current fused language/statistics soft-shrink prior as a SOTA candidate**
  - Attempt: use deterministic activation-alignment language clues fused with
    statistical clues as soft-score priors for official causal-shrink DFR.
  - Result: compact two-retrain screens showed a small fused lift over
    stats/heuristic/random controls, but the default 20-retrain fused checks
    tied across top-64 to top-512 at about `0.9315` test WGA.
  - Interpretation: the clue signal is real enough to keep as a diagnostic, but
    the current consumer does not clear the stronger
    `official_dfr_val_tr_retrains50` comparator around `0.9330`.
  - Action: do not promote this variant. Next work should add image/prototype
    clues or use fused priors inside a stronger objective rather than only
    conservative feature shrinkage.

- **Using source identity alone as proof of clue superiority**
  - Attempt: add language, image/prototype, and fused score sources, then run
    top-k downstream screens through the stronger soft-score `causal_dfr`
    objective.
  - Result: the stronger objective reached about `0.9401` test WGA and cleared
    the active local DFR comparator, but stats, language, image, fused, and
    heuristic top-k variants tied at the same WGA in the first single-seed
    screen.
  - Interpretation: the objective upgrade looked promising before seed checks,
    but source-specific ranking superiority is not yet established by
    downstream metrics even though ablation tables show distinct top-k sets.
  - Action: keep source-ablation overlap and confidence metrics in the loop,
    but require seed-stable downstream separation before claiming one clue
    source is better than another.

- **Pruning discovery soft scores to top-k support as an immediate improvement**
  - Attempt: make `discovery_score_top_k` affect soft-score consumers by
    zeroing `causal_feature_scores` outside the selected support, then rerun
    the clue-source screen through soft-score `causal_dfr`.
  - Result: candidate rankings separated slightly, but the source-score variants
    dropped below the full-score soft-prior result. Heuristic top-k remained
    about `0.9401` test WGA, while stats/language/fused were mostly around
    `0.9392` and fused top-128 fell to about `0.9388`.
  - Interpretation: top-k pruning is useful for diagnostics but too lossy as a
    default prior for the current objective.
  - Action: keep `discovery_score_soft_selection: selected` and
    `--prune-soft-scores` as opt-in diagnostics only; retain full soft scores
    for the current promotion path.

- **Raw soft-score causal DFR as a seed-stable promotion candidate**
  - Attempt: run paired seeds 101/102/103 for official DFR versus full-score
    soft `causal_dfr` fused, heuristic, and random top-64 variants.
  - Result: fused top-64 beat baseline on seed 101 by about `+0.0087` test WGA,
    but lost on seeds 102 and 103 by about `-0.0295` and `-0.0561`; mean test
    WGA was about `0.9058` versus official DFR at about `0.9315`.
  - Interpretation: the single-seed result was optimizer/seed fragile and
    cannot support promotion.
  - Action: require `scripts/run_waterbirds_clue_seed_stability.py` before any
    future clue-fusion promotion claim.

- **DFR retrain averaging as sufficient stabilization**
  - Attempt: add `dfr_num_retrains` and evaluate a 3-head averaged soft-score
    causal DFR ensemble.
  - Result: variance improved substantially, but mean fused top-64 test WGA was
    only about `0.9216`, below the official DFR baseline by about `0.0099`.
    LBFGS and a lighter nuisance penalty also failed exploratory screens.
  - Interpretation: retrain averaging is a useful diagnostic and variance
    reducer, but the current causal DFR head gives up too much mean WGA.
  - Action: keep retrain averaging available, but move the clue priors into a
    stronger official-DFR-compatible objective rather than promoting this head.

- **Bridge scores as validation-split causal DFR soft priors**
  - Attempt: add a paired runner that feeds stats and bridge-fused score files
    into `causal_dfr_nuisance_prior: soft_scores`, then screen nuisance weights
    and retrain averaging against the official 50-retrain DFR comparator.
  - Result: seed 101 looked strong (`0.9401` WGA for top-512 at nuisance weight
    `10`), but seeds 102/103 collapsed; the three-seed mean was about `0.9060`.
    Five DFR retrains did not rescue the path and averaged about `0.9195`.
  - Interpretation: the validation-split causal DFR head is still too seed
    fragile or gives up too much mean WGA, even when bridge scores are used as
    the nuisance prior.
  - Action: keep the runner as a diagnostic harness, but do not promote this
    consumer. Prefer official-compatible shrink consumers unless a materially
    stronger causal DFR objective is implemented.

- **Naive offline clue value policy as a standalone ranker**
  - Attempt: train a ridge contextual-bandit value policy from replayed clue
    packet/action reward rows, then rank held-out fixture packets by predicted
    best action value.
  - Result: on the refreshed trace snapshot, alpha `10` raw policy top-1
    causal-target recovery was `0.25`, below the stats-margin baseline at
    `0.625`. It was non-random and useful at wider supports, but not a
    standalone replacement for stats or the current bridge-fused candidate.
  - Interpretation: the reward table is useful infrastructure, but simple
    scalar value regression over tiny logged traces overfits action/fixture
    artifacts and does not yet learn a general clue proposer.
  - Action: do not promote raw offline policy. Continue with conservative
    policy/stat fusion and upgrade training to pairwise/listwise or
    artifact-risk-aware objectives before downstream Waterbirds promotion.

- **Policy-fused scores as the next immediate Waterbirds promotion**
  - Attempt: expose offline policy scores as Waterbirds discovery-score sources
    and evaluate `policy_fused/w0.5/top512` through the official causal-shrink
    consumer with paired compact official/stat/random controls.
  - Result: the candidate beat official DFR and stats on mean over two compact
    seeds, but the margin was small and it was non-negative against the best
    deterministic random-score control on only `1/2` seeds. It also trailed the
    already active bridge-fused compact result.
  - Interpretation: policy learning is useful as an auxiliary scorer, but it is
    not yet a stronger primary mechanism than the existing bridge/stat fusion.
  - Action: do not run a full 50-retrain promotion for this variant. Use it as
    an ingredient for future bridge-policy hybrids or better rank objectives.

- **Constrained support optimizer as an immediate margin widener**
  - Attempt: construct top-512 support directly by preserving a stats core,
    filling with bridge-ranked features, and capping env-dominant additions.
  - Result: the loose constrained variant tied the incumbent bridge-fused
    compact result exactly, while stricter variants regressed and failed the
    best-random gate on one of two compact seeds.
  - Interpretation: constrained support selection is useful infrastructure, but
    fixed stats-core/env-cap rules either reconstruct the incumbent or remove
    too much bridge support.
  - Action: do not promote these fixed variants. Reuse the harness only if the
    constraint is driven by a learned artifact-risk head or active boundary
    tests.

- **First artifact-risk head as an immediate bridge margin widener**
  - Attempt: train a small risk head from replayed fixture traces and use it to
    penalize risky bridge-fused Waterbirds features globally or only near the
    top-512 boundary.
  - Result: after fixing intercept handling, the risk head produced nonzero
    estimates, but it did not alter the current `bridge_fused/w0.3/top512`
    support. Weight and boundary-window diagnostics kept `512/512` overlap with
    the incumbent support.
  - Interpretation: the incumbent already has very low env/artifact risk under
    this signal, so the first risk head mostly confirms the support audit rather
    than finding replacements.
  - Action: keep artifact-risk scoring as a guardrail/instrumentation path, but
    do not spend downstream benchmark budget on the current variants. Move to
    pairwise/listwise supervision or stronger active-boundary tests.

- **First pairwise bridge ranker as an immediate Waterbirds margin widener**
  - Attempt: train a pairwise ridge bridge ranker from within-run replay-trace
    comparisons, fuse it conservatively with stats, and evaluate the resulting
    `pairwise_bridge_fused` scores through the official causal-shrink
    Waterbirds consumer.
  - Result: held-out fixture diagnostics improved over the scalar bridge at
    top-1 and improved stats at top-2/top-4 when fused, but compact Waterbirds
    screens for weights `0.1`, `0.3`, and `0.5` all trailed stats and the best
    deterministic random control on mean.
  - Interpretation: pairwise supervision is a useful training/evaluation
    substrate, but the current trace targets and feature surface still do not
    identify better top-512 Waterbirds replacements.
  - Action: do not full-budget promote these pairwise-fused variants. Keep the
    evaluator and score source, and only revisit after adding stronger
    active-boundary tests or richer listwise/query-level supervision.

- **Conditional-signal active boundary as an immediate margin widener**
  - Attempt: retest only features near the `bridge_fused/w0.3/top512` cutoff
    with cheap conditional-signal checks, then rerank that local boundary while
    keeping the core support fixed.
  - Result: the method made real replacements (`51/512`) and reduced
    env-dominant selected features from `5` to `2`, but compact downstream WGA
    regressed below the incumbent and failed the stats/best-random gates on one
    of two seeds.
  - Interpretation: simply lowering env-dominant count or improving conditional
    signal is not sufficient; the replaced features likely include useful
    bridge support that the official causal-shrink consumer needs.
  - Action: keep active-boundary tooling, but do not promote the
    conditional-signal variant. A future boundary scorer should use model-effect
    or validation-loss-aware evidence before another downstream screen.

- **Single-probe model-effect active boundary as an immediate margin widener**
  - Attempt: fit a lightweight balanced train-split probe on incumbent support
    plus the local top-512 boundary, then rerank boundary features by held-out
    WGA/log-loss damage when each feature is ablated.
  - Result: the scorer made larger real replacements (`76/512`) and produced a
    strong seed-102 compact WGA (`0.9377`), but mean compact WGA still trailed
    the incumbent (`0.9349` versus `0.9353`) and it cleared stats/best-random
    gates on only `1/2` seeds.
  - Interpretation: model-effect evidence is closer to the downstream consumer
    than conditional signal, but a single train-split probe is too noisy and can
    admit env-dominant replacements that hurt seed stability.
  - Action: keep the scorer as instrumentation. Do not full-budget promote it;
    revisit only with split-ensembled or paired replacement evaluation.

- **Split-ensembled model-effect active boundary as a stability fix**
  - Attempt: average boundary ablation evidence across five balanced probe
    splits before reranking the `bridge_fused/w0.3/top512` boundary.
  - Result: the ensemble selected the same top-512 support as the single-probe
    model-effect scorer and reproduced the same compact downstream metrics:
    mean WGA `0.9349`, one negative stats/best-random seed, and mean below the
    incumbent `0.9353`.
  - Interpretation: the failure is not primarily due to one unlucky train/probe
    split; the current ablation target itself admits replacements that are not
    stable under the official causal-shrink consumer.
  - Action: stop split-ensembling this exact scorer. Next boundary work should
    use paired replacement evaluation or add an explicit env-risk constraint.

- **Env-guarded model-effect active boundary as a promotion candidate**
  - Attempt: subtract shortcut-risk evidence from the model-effect boundary
    scorer so boundary replacements are both downstream-aware and constrained
    against env-dominant features.
  - Result: support diagnostics and compact screening improved: only `5/512`
    support replacements, env-dominant selected features `5 -> 4`, compact mean
    WGA `0.9355` versus incumbent `0.9353`, and all compact gates cleared.
    However, the five-seed 50-retrain promotion screen averaged `0.9358`, below
    the incumbent `0.9368`, and failed stats/best-random on seed 104.
  - Interpretation: the env guard is a useful boundary guardrail, but the fixed
    correlation penalty is not enough to choose replacements that generalize
    across the full paired gate.
  - Action: do not promote this variant. Keep it as the strongest boundary
    diagnostic so far; next work should score paired replacements directly or
    learn the guard from downstream paired outcomes.

- **Activation-gap fields as bridge-ranker features**
  - Attempt: add activation label/environment gaps and alignment one-hot fields
    to bridge training rows and the ridge ranker, then refresh the offline
    fixture traces.
  - Result: leave-one-fixture-out causal-target recovery regressed. Bridge
    top-1 fell to `0.25` while stats top-1 was `0.625`, and larger ridge-alpha
    values did not recover the old behavior.
  - Interpretation: these activation fields are too confounded in the tiny
    fixture training corpus, or the refreshed trace corpus changed enough that
    the earlier held-out bridge comparison is no longer apples-to-apples.
  - Action: backed out the source feature expansion. Future bridge supervision
    should be versioned with its trace corpus and evaluated before downstream
    scoring.

- **Official DFR soft clue shrink as an immediate promotion**
  - Attempt: add an official-DFR-compatible soft-score shrink config and run a
    paired 3-seed fused top-64 pruned clue screen.
  - Result: the tuner selected nontrivial clue shrink values (`0.7`, `0.7`,
    `0.8`) and produced low variance, but mean test WGA was about `0.9311`
    versus official DFR at about `0.9315`; seed deltas were approximately
    `+0.0002`, `-0.0029`, and `+0.0016`.
  - Interpretation: this is not promotable yet, but it is a near-miss in the
    right training path rather than a collapse.
  - Action: keep the config as the next official-compatible baseline and tune
    clue support/scale grids under the paired seed gate.

- **Uniform conflict-example upweighting as the first upstream feature fix**
  - Attempt: use the new backbone fine-tuning sample modes to upweight
    Waterbirds train examples whose bird label and background conflict, then
    evaluate the resulting limit384 feature artifacts through unchanged
    official DFR.
  - Result: conflict-only e3 at weight `3.0` reached downstream official DFR
    test WGA `0.84375`, tying the old e5 no-conflict limit384 anchor but not
    exceeding it. Conflict-only e5 fell to `0.75`; lighter e3 conflict weight
    `1.5` fell to `0.8125`; grouped-conflict weight `3.0` collapsed at base ERM
    for e3 with test WGA `0.0`.
  - Interpretation: simple uniform conflict oversampling changes the feature
    representation, but the useful region is narrow and not strong enough to
    justify full-run compute.
  - Action: do not launch full runs for these exact conflict-only or
    grouped-conflict settings. Future upstream work should use a different
    backbone/source or a staged/tempered sampling schedule rather than static
    conflict weighting from the first epoch.

- **Simple staged conflict upweighting as a stronger upstream feature fix**
  - Attempt: start Waterbirds ERM backbone fine-tuning with natural sampling,
    then switch to conflict-example upweighting after two epochs on the limit384
    diagnostic slice.
  - Result: e5/LR `0.001`, conflict weight `3.0`, sample warmup `2`, and seed
    `101` produced strong base ERM test WGA `0.9375`, but downstream official
    DFR test WGA was only `0.8125`.
  - Interpretation: better base classifier WGA is not sufficient; the exported
    representation still did not help the official DFR head and underperformed
    the old no-conflict e5 limit384 anchor (`0.84375`).
  - Action: do not launch full runs for this staged conflict recipe. Treat
    simple conflict-sampling schedules as locally exhausted unless a future
    mechanism changes the representation objective or feature source more
    substantially.

## 2026-05-01

- **Simple support filtering around the refreshed bridge-fused candidate**
  - Attempt: use support diagnostics from `bridge_fused/w0.3/top512` to create
    official-shrink score variants that demote env-dominant bridge-selected
    features or reshape the bridge score before discovery-score ingestion.
    Variants included `env_filter`, `margin_gate`, `stats_fill`,
    `soft_env_penalty`, `stats_anchor`, `score_sqrt`, and `score_square`.
  - Result: hard `env_filter` and `stats_fill` looked slightly better than the
    incumbent on compact seeds `101`/`102`, but `env_filter` failed the full
    five-seed 50-retrain promotion gate. It reached mean WGA `0.9361370683`,
    below incumbent `0.9367601395`, and was non-negative against the best random
    control on only `3/5` seeds. `margin_gate` and `stats_anchor` regressed;
    `score_sqrt` and `score_square` tied the compact incumbent exactly.
  - Interpretation: the incumbent already avoids most env-dominant features;
    simple post-hoc filtering over-prunes useful support, and rank-preserving
    score transforms do not meaningfully alter the current official-shrink
    selection path.
  - Action: keep the support-variant tooling for diagnostics, but do not spend
    more full-budget compute on these local variants. Move next to better bridge
    supervision or a materially different official-compatible consumer.

- **Active latent patch-probe stack as a Waterbirds improvement path**
  - Attempt: build a trainable patch-level counterfactual probe over frozen
    DINO hidden states and official-DFR component features. Variants included a
    single `PatchFlipProbe`, additive CLS/token-norm priors inspired by
    Distribution Transformers, multi-hypothesis patch-mask posteriors,
    best-of-K/effect-best objectives, scalar feature-score exports, and
    intervention-derived feature tables such as original+edited and delta views.
  - Result: the probe produced stronger frozen-head intervention diagnostics
    than passive selectors. On the limit384 zero-replacement diagnostic, the
    best-of-K 4-component effect-selected mask reached about `0.304` mean
    decision-logit drop versus about `0.269` for a single learned mask and
    `0.239` for CLS-top. However, downstream consumers repeatedly failed:
    feature scores tied random controls, direct effect objectives overfit the
    proxy, pure deltas hurt, and the only `original_plus_edited` WGA lift was a
    one-test-example seed-101 artifact (`0.90625`, `0.875`, `0.875` over seeds
    `101`-`103`). Root-cause checks found no row-order or concatenation bug; the
    lift required the generated seed-101 edited table plus DFR `C=0.7`, while
    fixed `C=1.0` returned to `0.875`.
  - Interpretation: patch probing was useful as a diagnostic of frozen-head
    decision-sensitive regions, but the current mechanisms did not transfer
    into seed-stable downstream Waterbirds WGA. The compact WGA slice is also
    one-example sensitive, making it easy to mistake artifacts for progress.
  - Action: remove the patch-probe runner, intervention helpers, and tests from
    active code. Preserve only the component-aware feature export path. Future
    work should restart from a principled objective and must cross generated
    artifact seed, DFR fit seed, fixed-C controls, and random controls before
    treating compact lifts as meaningful.

- **Global supervised contrastive loss as an immediate upstream feature fix**
  - Attempt: add a global cross-background supervised-contrastive loss during
    Waterbirds ResNet feature generation, with same-label/different-background
    positives and same-background/different-label hard negatives, then evaluate
    exported penultimate features through unchanged official DFR.
  - Result: seeded limit384 e5/LR `0.001` diagnostics at contrastive
    `w=0.05,t=0.2` and `w=0.2,t=0.15` both matched the seeded no-contrastive
    control at official DFR test WGA `0.71875`. The feature matrices moved
    measurably, but base WGA and downstream DFR metrics did not improve.
  - Interpretation: global label-level contrast is too coarse for the current
    Waterbirds representation problem; it may align birds across backgrounds,
    but it does not expose a better DFR-ready bird/background factorization.
  - Action: keep the implementation and seed control as infrastructure, but do
    not launch full runs for these global supervised-contrastive settings. Move
    next to patch/object/component decomposition before applying DFR or clue
    priors.

- **Fixed DINO patch center/background pooling as a full benchmark candidate**
  - Attempt: use frozen local DINOv2-small patch tokens pooled into CLS,
    center-patch, corner-background, and center-minus-background components,
    then evaluate through unchanged `official_dfr_val_tr_retrains50`.
  - Result: the limit384 diagnostic was encouraging at `0.875` official DFR
    test WGA, but the no-limit seed101 run reached only
    `0.9112149477005005` test WGA and `0.9368311762809753` accuracy.
  - Interpretation: fixed center/corner patch pooling is too crude to beat the
    official full-data comparator even though it improves the small diagnostic
    slice.
  - Action: do not promote this exact patch-component recipe. Keep DINO
    decomposition as a direction, but use a better component selector or an
    efficient crop/object extraction path before spending seed-sweep compute.

- **CLS-similarity and token-norm DINO patch selectors as immediate upgrades**
  - Attempt: replace fixed center/corner patch pooling with frozen DINOv2-small
    selector components, using CLS-similarity top/bottom patches or token-norm
    top/bottom patches, then evaluate through official DFR on the limit384
    diagnostic slice.
  - Result: CLS-similarity tied the fixed patch diagnostic at test WGA `0.875`;
    token norm reached only `0.84375`. A compact clue/soft-shrink pass on the
    CLS selector also stayed at `0.875`, with fused clues changing the selected
    feature support but not the downstream metric.
  - Interpretation: selector pooling changes the representation enough to be a
    useful diagnostic, but the current pooled summaries and soft-shrink consumer
    do not create a benchmark improvement.
  - Action: do not launch full no-limit runs for these selector-pooling recipes
    alone. Use the latent patch intervention path to test whether selected
    patches have label-specific counterfactual effects that beat random,
    background-like, donor, and prototype controls.

- **Fixture-level ceiling as evidence of SOTA**
  - Attempt: compose counterfactual augmentation with adversarial probe training
    and evaluate first on the Waterbirds-style fixture.
  - Result: the new `counterfactual_adversarial` method reached WGA/accuracy
    1.0 and improved probe selectivity, but the fixture is too small and too
    synthetic to support a literature claim.
  - Interpretation: reaching the fixture ceiling is useful as a regression and
    mechanism check, but it does not establish progress over published
    Waterbirds results.
  - Action: treat this as a candidate method, not a result claim. Move next to
    the real local feature-table adapter and compare against published WGA under
    matching assumptions.

## 2026-04-23

- **Stronger nuisance adversary as a direct WGA improvement on real Waterbirds**
  - Attempt: strengthen `counterfactual_adversarial` on the real Waterbirds
    feature benchmark with deeper nuisance heads, later warmup, detached
    nuisance-head updates, and lower nuisance-loss weights.
  - Result: stronger-head variants improved representation probes, with the
    detached-head late-warmup run reaching causal probe `0.924`, nuisance probe
    `0.586`, and selectivity `0.338`, but its test WGA stayed lower at `0.752`
    than the historical best simple scheduled run at `0.782`.
  - Interpretation: stronger adversaries help suppress nuisance information,
    but on the current setup they still trade away too much worst-group
    performance. Better probe selectivity is not automatically a better
    benchmark result.
  - Correction: a later rerun of the simple scheduled variant exposed a
    nuisance-head optimizer bug in `fit_counterfactual_adversarial()`, so the
    comparison against the live baseline was temporarily invalid. After fixing
    that bug, the repaired simple scheduled run recovered to `0.687` WGA and
    `0.909` accuracy, still below the historical `0.782` run and below the best
    stronger-head variants.
  - Action: do not treat the simpler scheduled variant as the uncontested live
    benchmark-facing best method. The practical frontier is now between the
    repaired simple schedule and the stronger-head schedule variants that still
    reach `0.752` to `0.765` WGA.

- **Explicit-only discovery-mask learner as a drop-in replacement for the heuristic Waterbirds mask**
  - Attempt: retrain the discovery scorer using only explicit supervision from
    synthetic and tiny fixture datasets, exclude `waterbirds_features` from
    supervision entirely, and add a pairwise ranking loss so the scorer learns
    within-dataset feature ordering rather than only pointwise probabilities.
  - Result: the revised learner trained on only `28` explicit rows across
    `synthetic_linear`, `synthetic_nonlinear`, `dsprites_tiny`, and
    `causal3d_tiny`. Its top-512 Waterbirds mask overlapped the heuristic mask
    on `205` features, with precision `0.400`, recall `0.424`, and Jaccard
    `0.259`. The first downstream benchmark for
    `waterbirds_features_counterfactual_adversarial_discovery_mask` landed at
    test WGA `0.522` and accuracy `0.882`.
  - Correction: that `0.522` result was later invalidated as a discovery
    failure because the shared counterfactual-adversarial trainer had a frozen
    nuisance-head bug. After fixing the optimizer path, the direct learned-mask
    benchmark reran to test WGA `0.687` and accuracy `0.909`, exactly matching
    the repaired heuristic schedule baseline.
  - Interpretation: removing heuristic self-distillation fixed the supervision
    story, but the current evidence still does not show a learned-mask win over
    the repaired heuristic baseline. What failed was the earlier conclusion,
    not necessarily the learned-mask idea itself.
  - Action: do not keep citing the pre-fix `0.522` run as evidence against the
    learned mask. Future discovery work should compare against the repaired
    baseline and the stronger-head schedule variants, and should keep using
    richer clue targets or structured ranking objectives tied to downstream
    intervention quality.

## 2026-04-26

- **More stage-1 selector weighting as the next grouped-gate breakthrough**
  - Attempt: extend the grouped instability-JTT selector family with two more
    principled stage-1 scores: counterfactual excess label loss
    (`counterfactual_loss_increase_mean`) and a group-weighted version of that
    same signal
    (`group_loss_weighted_counterfactual_loss_increase_mean`).
  - Result: the excess-loss selector reached compact test WGA `0.667`, val WGA
    `0.647`, and promotion score `0.647` with zero promotable rows. The
    group-weighted excess-loss selector reached compact test WGA `0.673`, val
    WGA `0.639`, and promotion score `0.639`, also with zero promotable rows.
  - Interpretation: using label-loss degradation is cleaner than raw
    disagreement, but this branch still fails the stricter compact promotion
    rule and does not improve on grouped instability-JTT. Adding group failure
    weighting nudged test WGA up a little while making validation alignment
    worse.
  - Action: stop defaulting to more stage-1 weighting variants. The next
    mechanism step should target the stage-2 counterfactual-adversarial
    objective, such as group-conditional replay or hard-group consistency,
    rather than keep changing only the selector.

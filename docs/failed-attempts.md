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

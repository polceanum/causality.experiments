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

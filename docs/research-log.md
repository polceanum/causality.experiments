# Research Log

Chronological notes about experiments, implementation choices, and empirical
signals. Keep this focused on what was tried and what was learned.

## 2026-04-22

- Read the source document, which is a survey and experiment blueprint rather
  than a single fully specified algorithm. The practical research direction is
  to compare counterfactual interventions, invariance penalties, probing, and
  meta-causal generalization under shared evaluation scaffolding.
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

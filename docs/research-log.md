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

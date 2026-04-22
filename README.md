# causality.experiments

Dataset-driven experiments for causal extraction via counterfactual probing
and meta-causality. The repository starts with a lightweight harness that runs
on the existing `orpheus` conda environment without installing new packages.

## Quick start

```bash
conda run -n orpheus python -m causality_experiments run \
  --config configs/experiments/01_synthetic_linear.yaml
```

Run every tiny fixture experiment:

```bash
conda run -n orpheus python scripts/run_all_fixtures.py
```

Generate fixture mirror files:

```bash
conda run -n orpheus python -m causality_experiments make-fixtures
```

Summarize runs:

```bash
conda run -n orpheus python -m causality_experiments summarize --runs outputs/runs
```

## Shape of the implementation

- `causality_experiments.data` provides dataset adapters and tiny generated
  mirrors for all 8 experiments described in the source document.
- `causality_experiments.methods` provides runnable `constant`, `oracle`, and
  `erm`, `irm`, and `counterfactual_augmentation` baselines plus adapter
  contracts for causal probes, β-VAE/iVAE/CITRIS/CSML/DeepIV.
- `causality_experiments.metrics` records accuracy, worst-group accuracy,
  support recovery, and an ATE-style proxy where ground truth supports it.
- `configs/experiments` contains one runnable fixture config per experiment.

The heavier causal methods are intentionally explicit adapter stubs in this
first pass. They can be implemented behind the same fit/predict interface
without changing datasets, metrics, or run outputs.

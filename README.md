# causality.experiments

Dataset-driven experiments for causal extraction via counterfactual probing
and meta-causality. The repository starts with a lightweight harness that runs
on the existing `orpheus` conda environment without installing new packages.

## Operating constraints

- Everything should run locally on the Mac notebook in the `orpheus` conda
  environment.
- Do not rely on paid LLM APIs, hosted model APIs, cloud processing, or external
  services for core experiments.
- Do not replace the existing Mac-specific PyTorch install.
- Larger benchmark integrations should always keep a tiny local fixture path
  and a modest smoke-test path.
- Prefer small PyTorch models, seed sweeps, and careful metrics over
  compute-heavy architectures that cannot run locally in a reasonable time.
- CI is only a regression guardrail. It should run lightweight tests and smoke
  checks, not full research sweeps.

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

Write a Markdown research report:

```bash
conda run -n orpheus python scripts/write_research_report.py
```

Run a seed sweep and report mean/std:

```bash
conda run -n orpheus python scripts/run_seed_sweep.py --match 07 --seeds 11,12,13
conda run -n orpheus python scripts/report_seed_sweep.py --match 07
```

Report causal/nuisance probe diagnostics:

```bash
conda run -n orpheus python scripts/report_probe_diagnostics.py --match 05_waterbirds
```

Check benchmark/literature alignment:

```bash
conda run -n orpheus python scripts/report_benchmark_alignment.py
```

Run a real-benchmark-compatible Waterbirds feature table:

```bash
conda run -n orpheus python -m causality_experiments run \
  --config configs/benchmarks/waterbirds_features.yaml
```

## Shape of the implementation

- `causality_experiments.data` provides dataset adapters and tiny generated
  mirrors for all 8 experiments described in the source document.
- `causality_experiments.methods` provides runnable `constant`, `oracle`,
  `erm`, `group_balanced_erm`, `group_dro`, `irm`, `jtt`,
  `adversarial_probe`, `counterfactual_adversarial`, and
  `counterfactual_augmentation` baselines plus adapter contracts for causal
  probes, β-VAE/iVAE/CITRIS/CSML/DeepIV.
- `causality_experiments.metrics` records accuracy, worst-group accuracy,
  support recovery, and an ATE-style proxy where ground truth supports it.
- `configs/experiments` contains one runnable fixture config per experiment.

The heavier causal methods are intentionally explicit adapter stubs in this
first pass. They can be implemented behind the same fit/predict interface
without changing datasets, metrics, or run outputs.

## Project logs

- [Research log](docs/research-log.md)
- [Failed attempts log](docs/failed-attempts.md)
- [Current state and plan](docs/current-state.md)
- [Literature context](docs/literature-context.md)
- [Benchmark adapters](docs/benchmark-adapters.md)

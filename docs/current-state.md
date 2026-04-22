# Current State and Plan

This is the research handoff/state file. Update it after meaningful
implementation or experiment rounds. Keep it focused on the causal-extraction
goal, not incidental development mechanics.

## Current State

- `main` is pushed to GitHub.
- Latest pushed commits:
  - `f36eb63` `Implement causal experiment harness`
  - `3d171d7` `Add causal method sweeps`
  - `20f547c` `Add experiment tracking logs`
- The repo contains a runnable PyTorch experiment harness for all 8 paper
  experiments using tiny generated fixtures.
- Runnable methods:
  - `constant`
  - `oracle`
  - `erm`
  - `irm`
  - `counterfactual_augmentation`
- Sequence fixtures now use integer tokens, an embedding-pooling classifier,
  and token-specific counterfactual augmentation.
- Seed-sweep scripts are available and have been run on `07_text_toy` and
  `08_fewshot_ner`; robust methods improve mean WGA but with high variance.
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
  - `conda run -n orpheus python -m causality_experiments summarize --runs outputs/runs`

## Near-Term Plan

1. Start filling one adapter stub with a concrete causal probing baseline.
   - Train a probe on learned hidden representations for fixture concepts.
   - Add completeness/selectivity metrics where fixture metadata supports it.
2. Add stronger literature baselines.
   - Group-balanced ERM or GroupDRO for known-group settings.
   - JTT-style two-stage reweighting.
3. Add real-data adapter documentation.
   - Specify expected local file formats for Waterbirds, dSprites/3DShapes,
     Causal3DIdent, and NER before implementing dataset-specific loaders.

## Current Risks

- Fixture tasks are useful for harness iteration but are not substitutes for
  real benchmark results.
- Current sequence experiments are under-modeled.
- IRM remains sensitive to penalty schedules and may need config sweeps per
  dataset.
- ATE proxy is a rough diagnostic, not a validated causal-effect estimator.
- Sequence WGA improvements are promising but high-variance across seeds.

## Log Hygiene

- `docs/research-log.md` should track research progress and empirical signals.
- `docs/failed-attempts.md` should track failed or weak research/modeling
  attempts, not tooling friction.
- `docs/current-state.md` should stay as a compact handoff: current capability,
  next plan, risks, and active assumptions.

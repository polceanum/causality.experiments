# Current State and Plan

This is the handoff/state file. Update it after meaningful implementation or
experiment rounds.

## Current State

- `main` is pushed to GitHub.
- Latest pushed commits:
  - `f36eb63` `Implement causal experiment harness`
  - `3d171d7` `Add causal method sweeps`
- Worktree was clean before these docs were added.
- The repo contains a runnable PyTorch experiment harness for all 8 paper
  experiments using tiny generated fixtures.
- Runnable methods:
  - `constant`
  - `oracle`
  - `erm`
  - `irm`
  - `counterfactual_augmentation`
- Adapter stubs still intentionally remain for heavier methods:
  causal probes, beta-VAE/iVAE, CITRIS, CSML, and DeepIV.
- Core commands:
  - `conda run -n orpheus python -m pytest`
  - `conda run -n orpheus python scripts/run_all_fixtures.py`
  - `conda run -n orpheus python scripts/run_method_sweep.py`
  - `conda run -n orpheus python scripts/report_best_methods.py`
  - `conda run -n orpheus python -m causality_experiments summarize --runs outputs/runs`

## Near-Term Plan

1. Improve the text/sequence track.
   - Replace the current float-token MLP treatment with a small embedding-based
     sequence model.
   - Add counterfactual token-flip augmentation that preserves causal tokens and
     randomizes confounder tokens.
   - Re-run `07_text_toy` and `08_fewshot_ner` sweeps.
2. Make reports more research-friendly.
   - Add a compact Markdown report writer that records best method per
     experiment, key metrics, config path, and run directory.
   - Append report highlights into `docs/research-log.md`.
3. Start filling one adapter stub with a concrete causal probing baseline.
   - Train a probe on learned hidden representations for fixture concepts.
   - Add completeness/selectivity metrics where fixture metadata supports it.
4. Add real-data adapter documentation.
   - Specify expected local file formats for Waterbirds, dSprites/3DShapes,
     Causal3DIdent, and NER before implementing dataset-specific loaders.

## Current Risks

- Fixture tasks are useful for harness iteration but are not substitutes for
  real benchmark results.
- Current sequence experiments are under-modeled.
- IRM remains sensitive to penalty schedules and may need config sweeps per
  dataset.
- ATE proxy is a rough diagnostic, not a validated causal-effect estimator.

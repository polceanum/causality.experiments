# Literature Context

Use this file to keep result reporting honest. Fixture results are useful for
iteration, but they are not directly comparable to published benchmark numbers.

## Reporting Rule

- Every research result should say whether it is from a tiny fixture, a local
  synthetic benchmark, or a real benchmark adapter.
- Tiny fixture numbers must not be described as state of the art.
- For real benchmarks, reports should compare against published ERM, robust
  baseline, and best-known/SOTA numbers where available.
- Before making a serious claim, refresh this file with current papers or
  benchmark tables and cite sources.

## Waterbirds / Spurious Correlation Robustness

The repo currently has `05_waterbirds`, a tiny Waterbirds-style fixture. It is
not the real Waterbirds benchmark, so its WGA/accuracy values are only harness
diagnostics.

Reference points for the real Waterbirds benchmark:

- JTT reports Waterbirds test worst-group accuracy around `86.7` and average
  accuracy around `93.6`.
  Source: Liu et al., "Just Train Twice: Improving Group Robustness without
  Training Group Information", ICML 2021,
  https://proceedings.mlr.press/v139/liu21f.html
- JTT paper states it closes a large fraction of the gap between ERM and
  GroupDRO across Waterbirds, CelebA, MultiNLI, and CivilComments-WILDS.
  Source: https://proceedings.mlr.press/v139/liu21f/liu21f.pdf
- A recent group-robust classification paper reports Waterbirds reference
  numbers including GroupDRO around `91.4` worst-group accuracy and other
  methods around the high-80s to low-90s, depending on assumptions and
  validation access.
  Source: https://openreview.net/pdf?id=2OcNWFHFpk

Implication for this repo:

- Our tiny Waterbirds-style fixture reaching WGA near `1.0` is not meaningful as
  a SOTA comparison.
- A meaningful comparison requires a real Waterbirds adapter, matching splits,
  matching pretrained/backbone assumptions, and a local feasible model setup.
- Until then, report `05_waterbirds` as "Waterbirds-style fixture" only.

## Sequence Fixtures

`07_text_toy` and `08_fewshot_ner` are synthetic/local fixtures, not real NLP
benchmarks. Their purpose is to stress causal token/confounder behavior.

Reference targets still need to be identified before claims:

- For few-shot NER, compare against the specific benchmark/source dataset once
  a real local adapter exists.
- For synthetic token tasks, compare against internal baselines and report
  ablations; do not compare to unrelated NLP leaderboard numbers.

## Causal Representation / Factor Benchmarks

`03_dsprites_3dshapes`, `04_causal3dident`, and `06_shapes_spurious` are tiny
factor fixtures. They need real adapters before comparing to published
disentanglement or causal-representation numbers.

Expected future comparison families:

- beta-VAE / FactorVAE-style disentanglement baselines.
- iVAE / nonlinear ICA with auxiliary variables.
- CITRIS/Causal3D-style causal representation methods.
- Metrics: MIG, DCI, IRS, graph support recovery, and intervention robustness,
  where the real benchmark supports them.

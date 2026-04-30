# Benchmark Adapters

Real benchmark adapters should be local-only and modest enough to smoke-test on
the Mac notebook. They should also preserve enough benchmark metadata for fair
comparison to literature numbers.

## Waterbirds Features

Config template:

```bash
conda run -n orpheus python -m causality_experiments run \
  --config configs/benchmarks/waterbirds_features.yaml
```

Expected local CSV path by default:

```text
data/waterbirds/features.csv
```

Required columns:

- `split` or `fold`: values must include `train`, `val`, and `test`.
- `y`, `label`, `target`, or `bird_label`: binary bird label.
- `place`, `background`, `env`, or `spurious`: binary spurious/background
  attribute.

Optional columns:

- `group` or `group_id`: explicit group ID. If omitted, the adapter uses
  `group = 2 * environment + label`.

Feature columns:

- Preferred: columns named `feature_0`, `feature_1`, ...
- Also accepted: columns starting with `x`.
- Fallback: all remaining numeric columns not used as metadata.

Optional causal mask for counterfactual methods:

- Add `dataset.causal_feature_columns` to list feature columns that should be
  preserved under counterfactual nuisance swaps.
- Or add `dataset.causal_feature_prefixes` for grouped features, for example
  `bird_`.
- Or set `dataset.causal_mask_strategy: label_minus_env_correlation` to derive
  a mask from the train split by keeping features whose absolute correlation
  with the label exceeds their absolute correlation with the environment. Use
  `dataset.causal_mask_min_margin` and optional `dataset.causal_mask_top_k` to
  control the selection.
- Or set `dataset.causal_mask_strategy: discovery_scores` with
  `dataset.discovery_scores_path` pointing at a score CSV containing
  `feature_name` and `score`. This is the preferred path for clue-fusion
  artifacts because it preserves soft `causal_feature_scores` for consumers
  such as soft-score `causal_dfr` and official causal-shrink DFR.
- Without this mask, methods such as `counterfactual_adversarial` are
  intentionally unavailable on real feature tables.

Literature comparability requirements:

- Use real Waterbirds train/validation/test splits.
- Report test worst-group accuracy.
- Document the feature extractor or backbone used to create the features.
- Only mark results as comparable when split semantics, labels, groups, and
  evaluation match the cited benchmark setup.

Benchmark provenance fields:

- Set `benchmark.provenance.feature_extractor` to the exact encoder or backbone
  used to generate the local feature table.
- Set `benchmark.provenance.feature_source` to the origin of the features,
  such as the local export process, checkpoint, or mirror description.
- Keep `benchmark.provenance.split_definition` aligned with the actual split
  semantics used by the CSV.
- If these fields are incomplete, the benchmark comparison reports now mark the
  Waterbirds comparison as blocked even if the local CSV is present.

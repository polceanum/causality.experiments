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

Literature comparability requirements:

- Use real Waterbirds train/validation/test splits.
- Report test worst-group accuracy.
- Document the feature extractor or backbone used to create the features.
- Only mark results as comparable when split semantics, labels, groups, and
  evaluation match the cited benchmark setup.

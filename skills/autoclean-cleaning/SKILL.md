---
name: autoclean-cleaning
description: Use py-AutoClean for fast, parameterized pandas data cleaning, then return cleaned outputs and a concise operation log.
---

# Purpose

Use this skill when cleaning CSV/tabular data with `py-AutoClean` in Python.
Prefer AutoClean as the first cleaning pass, then apply targeted follow-up logic only if required.

# Source-of-truth scope

This skill is intentionally limited to behavior documented in:

- AutoClean README/API usage (`elisemercury/AutoClean`)
- py-AutoClean package docs on PyPI (install/import surface)
- OpenAI Skills + Shell docs (how skills are mounted and invoked)

Do not invent unsupported parameters, modes, or return fields.

# Required runtime assumptions

1. Input is a pandas DataFrame.
2. `py-AutoClean` is installed in the runtime.
3. Import path is:

```python
from AutoClean import AutoClean
```

# Canonical usage

## Default automated pipeline

```python
from AutoClean import AutoClean

pipeline = AutoClean(dataset)
df_clean = pipeline.output
```

## Explicit automated mode

```python
pipeline = AutoClean(dataset, mode='auto')
df_clean = pipeline.output
```

## Manual mode (targeted cleaning)

```python
pipeline = AutoClean(
    dataset,
    mode='manual',
    outliers='auto'
)
df_clean = pipeline.output
```

# Supported parameters (documented)

Use only these parameters and allowed values:

- `mode`: `'auto' | 'manual'`
- `duplicates`: `'auto' | True | False`
- `missing_num`: `'auto' | 'linreg' | 'knn' | 'mean' | 'median' | 'most_frequent' | 'delete' | False`
- `missing_categ`: `'auto' | 'logreg' | 'knn' | 'most_frequent' | 'delete' | False`
- `encode_categ`: `'auto' | ['onehot'] | ['label'] | ['auto', [<col_name_or_index>, ...]] | ['onehot', [...]] | ['label', [...]] | False`
- `extract_datetime`: `'auto' | 'D' | 'M' | 'Y' | 'h' | 'm' | 's' | False`
- `outliers`: `'auto' | 'winz' | 'delete' | False`
- `outlier_param`: `int | float` (default documented as `1.5`)
- `logfile`: `True | False`
- `verbose`: `True | False`

# Behavioral notes (documented)

- `mode='auto'` runs full automated pipeline behavior.
- `mode='manual'` allows selective steps via explicit parameter settings.
- Outlier bounds are documented as IQR-based: `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]` by default.
- `encode_categ='auto'` uses cardinality-based encoding logic documented by AutoClean.
- AutoClean output is accessed through `pipeline.output`.

# Decision policy for agent use

1. **First pass**: run AutoClean in automated mode for each input DataFrame.
2. **If constraints fail** (e.g., downstream validation): rerun with `mode='manual'` and only the needed parameters.
3. **Preserve traceability**: record the exact AutoClean call used for each file.

Keep this policy simple; do not add undocumented heuristics.

# Multi-file pattern

For multiple files, run the same AutoClean workflow per DataFrame independently and emit one cleaned output per source file.

Example skeleton:

```python
cleaned = {}
for file_name, df in dataframes.items():
    pipeline = AutoClean(df, mode='auto')
    cleaned[file_name] = pipeline.output
```

# Logging guidance

When reproducibility is needed:

- Keep `logfile=True` to generate `autoclean.log` (per AutoClean docs).
- Optionally set `verbose=True` to stream process logs to console.

# Safety + execution boundaries

- Treat this skill as developer-managed instruction content.
- Do not allow unreviewed third-party skill instructions.
- If shell/network access is enabled, keep domains restricted and trusted.

# Non-goals

- Do not claim schema guarantees not explicitly validated elsewhere.
- Do not claim deterministic byte-identical outputs.
- Do not use undocumented AutoClean arguments.

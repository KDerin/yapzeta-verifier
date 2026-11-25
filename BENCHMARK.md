# Benchmark and evaluation

## What was available and rerun

The local project contained an existing regression script (`tests/test_verify.py`), its gold rows, a private project index, and a local OCRE catalog fixture. We copied these required files into an isolated validation directory and ran the script with `YZ_VERIFY_NLI=0` and `YZ_VERIFY_LLM=0`. Nothing was written to the original project tree. The private fixture and claim text are excluded from this package.

Observed output: 63 evaluated cases; supported precision 1.000; supported recall 0.958; three-class accuracy 0.968; zero known-false claims predicted supported; integration scenario passed. The script used the package’s `checks` compatibility module and legacy three-label outcomes (`DESTEKLİYOR`, `ÇELİŞİYOR`, `YETERSİZ`). It did not test the new six-state gate end-to-end.

## Recomputed metrics for the 63-case suite

Confusion matrix (rows = expected, columns = predicted):

| Expected \ Predicted | Supported | Contradicted | Insufficient |
|---|---:|---:|---:|
| Supported | 23 | 0 | 1 |
| Contradicted | 0 | 30 | 1 |
| Insufficient | 0 | 0 | 8 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Supported | 1.000 | 0.958 | 0.979 | 24 |
| Contradicted | 1.000 | 0.968 | 0.984 | 31 |
| Insufficient | 0.800 | 1.000 | 0.889 | 8 |

Three-class accuracy is 61/63 = 0.968. Macro-F1 is 0.950; micro-F1 equals accuracy (0.968). If `insufficient` is treated as abstention, selective coverage is (23+30)/63 = 84.13%, and abstention rate is 10/63 = 15.87%. If all three outcomes count as a decision, categorical coverage is 100%; this is not evidence of calibrated confidence.

The false-support denominator is 38 rows marked `bilinen_yanlis` by the existing suite, not all 63 cases and not all system outputs. False-support rate = false supports / known-false rows = 0/38 = 0%. Wilson score interval: two-sided 95% upper endpoint 9.18% (one-sided 95% upper endpoint 6.65%). The two-sided interval is reported as the default; the one-sided endpoint is provided to make the convention explicit. The interval reflects only a small, curated regression set and does not establish population performance.

## 702-claim benchmark: not verified

The raw 702-claim dataset, row-level labels, benchmark-generation protocol, and evaluation script were not found in the inspected local source/archive. Therefore we could not verify its provenance, label quality, split design, denominator, leakage controls, or reported 1.3% rate and 2.2% upper bound. These values are not reproduced and must not be cited as verified results. To reproduce them, provide the exact raw dataset, its label schema and adjudication protocol, the script/commit, and environment/version information.

For a false-support metric, disclose both numerator and denominator. A possible definition is false-supported decisions divided by all ground-truth unsupported claims; another is false-supported decisions divided by all claims the model labels supported. These answer different questions. The reported 1.3% cannot be interpreted until the original definition is known. Confidence interval method, confidence level, sidedness, and unit of resampling (claim vs. source/site) must also be specified.

## Leakage, memorization, and bias risks

The private regression suite is hand-authored, small, and tied to project-specific source strings and rules. Code constants and tests may share examples or domain vocabulary, creating contamination and overestimating generalization. Related claims may be near-duplicates; the suite does not document source-level or temporal holdout. The current metric is not an independent benchmark. The 702-case study cannot be assessed for train/test leakage, repeated source passages, labeler consistency, archaeological-period/site imbalance, language imbalance, or source-authority bias without its data and protocol.

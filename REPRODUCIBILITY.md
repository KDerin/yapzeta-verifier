# Reproducibility

## Public package tests

The maintainer reports a successful PyCharm run in Python 3.11.9 using NumPy 2.4.6 and pytest 9.2.0. The package installed into the isolated sibling environment `..\.release_venv`; `compileall` completed without errors. All four synthetic API tests passed (`4 passed in 0.09s`). This terminal output was supplied by the maintainer and was not independently rerun in this session. Tests:

- `test_insufficient_and_pending_conflict_paths`
- `test_missing_evidence_is_unverified`
- `test_six_states_and_display_mapping`
- `test_synthetic_support_and_contradiction`

To reproduce from the repository root using Python 3.11:

```bash
python -m pip install -e . pytest
python -m compileall -q yapzeta_verifier tests
python -m pytest -q
```

The RC2 archive is distributed with an external SHA-256 manifest covering its files and the ZIP. The packaging script is not part of the release.

These tests use synthetic examples and do not require a local index, secrets, or an NLI model. The verified run used `YZ_VERIFY_NLI=0` and `YZ_VERIFY_LLM=0`. CI is configured to run the package test suite on pushes and pull requests; the local result above is from PyCharm, not GitHub Actions.

## Existing private regression run

The existing project test was copied with only its required package, wrapper, test files, 65,757,195-byte local chunk index, and local OCRE lookup files into a separate validation-only folder. NLI and LLM judge were disabled. The script produced 63 rows and passed its gate (`0` false supports among 38 known-false examples, supported precision >= 0.9, recall >= 0.6, plus its integration check). The source tree was not changed. The private data and generated detailed report are not in this release candidate and should not be distributed. Re-running this test requires the same authorized private fixtures, which are intentionally not included.

## Benchmark not reproduced

The 702-claim data, labels, protocol, and original metric script were unavailable. This package cannot reproduce the 1.3% false-support rate or 2.2% upper bound.

## Environment notes

The validation run used Python 3.11.9 and NumPy 2.4.6. NLI was disabled (`YZ_VERIFY_NLI=0`); the LLM judge was disabled (`YZ_VERIFY_LLM=0`). The optional ONNX model was not present/used. Results can differ when enabled, so the published aggregate explicitly describes the deterministic run only.

## RC2 validation

The RC2 public synthetic suite was run in a separate Linux environment with Python 3.13.5, NLI and LLM disabled: 7 passed in 0.07s. Compilation completed without errors. This is a functional test, not an accuracy benchmark. The original 63-case private regression was not rerun for RC2.

## Final release validation

On 2026-11-25, all seven synthetic tests passed on Windows with Python 3.11.9 in 0.09 seconds. GitHub Actions runs 45525050953, 45525574071 and 45526350099 completed successfully. The core dependency audit reported no known vulnerabilities. These results do not establish scientific accuracy or production readiness.

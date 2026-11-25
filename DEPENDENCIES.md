# Dependency and license review

The candidate's core runtime dependency is NumPy. The optional NLI path imports ONNX Runtime and Hugging Face Tokenizers; neither the NLI model nor tokenizer files are bundled. Requests is used by the broader YAPZETA application but is not imported by this standalone verifier package and is excluded.

| Dependency | Candidate constraint | Role | License observed locally | Included in default install? |
|---|---|---|---|---|
| NumPy | `>=1.24,<3` | Numeric operations in deterministic/NLI checks | BSD-3-Clause plus bundled third-party notices (installed 2.4.6 metadata) | Yes |
| ONNX Runtime | `>=1.17,<2` | Optional local ONNX inference | MIT (installed 1.27.0 metadata) | No |
| Tokenizers | `>=0.15,<1` | Optional local tokenizer | Apache-2.0 classifier (installed 0.23.1 metadata) | No |
| pytest | `9.2.0` (maintainer-reported test environment) | Test runner only; not a runtime dependency | Not independently checked in this review | No |

This is a local metadata/license-file review of the installed environment, not a legal opinion or a current vulnerability-database audit. Runtime version constraints are compatibility ranges, not lockfile pins. The reported pytest version is the one used for the successful local test run. CI uses Python 3.11. The NLI model files may have separate licenses and must be checked independently before use or redistribution.

No license file was present in the inspected YAPZETA source root for the verifier package. The rights holder approved Apache-2.0 for the standalone source and original synthetic tests only. Third-party distribution rights still require review.

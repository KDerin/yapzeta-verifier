# YAPZETA Verifier (research preview)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24519190.svg)](https://doi.org/10.5281/zenodo.24519190)

**Version:** 0.1.0 · **Status:** research preview; not production validated

YAPZETA Verifier is a standalone, model-independent Python component for checking claim/evidence pairs against supplied text chunks. It combines deterministic domain checks, evidence quotation checks, source metadata rules, and an optional local natural-language-inference (NLI) signal. It does not fetch sources, build an index, call a language model, or include YAPZETA's main application.

**Technical author and developer:** Kartal Derin

## Scope and evidence

This preview contains only the verifier package and synthetic public API tests. The project’s existing private regression suite was run separately against a local copy of its private fixtures: 63 cases, 96.83% three-class accuracy, 1.000 precision for the supported class, 0 false supports among 38 labelled known-false cases, and one integration scenario passed. These are results for that specific legacy regression suite, not a representative benchmark or independent study. The 702-claim benchmark and its reported 1.3% false-support rate / 2.2% upper bound were not available for inspection and are **not verified or reproduced**. See [BENCHMARK.md](BENCHMARK.md).

## Install and run

Python 3.11 is used by the supplied CI workflow. Install the core dependency and run the public synthetic tests:

```bash
python -m pip install -e . pytest
python -m compileall -q yapzeta_verifier tests
python -m pytest -q
```

The maintainer reports a local run with Python 3.11.9, NumPy 2.4.6, and pytest 9.2.0: compilation completed without errors and all seven synthetic API tests passed (`7 passed in 0.09s`). See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for provenance and the earlier isolated regression results.

Minimal example:

```python
from yapzeta_verifier import verify_claims

chunks = [{
    "id": "example-1",
    "kaynak": "Example source, paragraph 1",
    "metin": "Stoa kat sayısı: 2.",
    "etiket": "Saha kaydı",
    "meta": {"source_type": "primary_excavation", "source_id": "example-source", "approval_status": "reported"},
}]
print(verify_claims(["Stoa iki katlıdır."], chunks))
```

The example is synthetic and does not assert a fact about any real site. The input fields `kaynak`, `metin`, and `etiket` retain the current Turkish API names.

## Six internal states

| State | Meaning in this implementation | User-facing label |
|---|---|---|
| `supported` | Rule checks support the claim and a source quote is found | `DESTEKLENİYOR` |
| `contradicted` | A deterministic check finds a conflict with the supplied source | `ÇELİŞİYOR` |
| `insufficient` | Supplied evidence does not establish the claim | `YETERSİZ_KANIT` |
| `conflict_pending` | The source marks an unresolved conflict | `YETERSİZ_KANIT` |
| `superseded` | The evidence is treated as historical/superseded | `ÇELİŞİYOR` |
| `unverified` | No usable source/quote/current chunk identity is verified | `YETERSİZ_KANIT` |

These are implementation labels, not calibrated probabilities or a claim that every edge case is correctly classified.

## Security and data boundary

Supply only data you are authorized to process. Do not place secrets, API keys, private indexes, unpublished research corpora, or personal data in public issues or test fixtures. No such material is included in this release candidate. See [SECURITY.md](SECURITY.md) and [LIMITATIONS.md](LIMITATIONS.md).

## License

Apache-2.0 applies to the standalone verifier source and original synthetic tests. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Other YAPZETA components, private data and third-party materials are outside this grant. NLI is opt-in (`YZ_VERIFY_NLI=1`); it is not validated by the public tests.

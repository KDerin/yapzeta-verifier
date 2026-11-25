# Limitations

- This is a research preview and has not been independently validated for production or academic decision-making.
- Deterministic, largely Turkish/domain-specific rules may be brittle under paraphrase, morphology, translation, OCR noise, or unfamiliar archaeological terminology.
- Lexical-overlap candidate selection can miss relevant evidence and rank misleading passages.
- Source type and approval metadata are supplied or inferred heuristically; the package cannot authenticate a source or validate a human approval event.
- A verbatim quote proves text presence, not that the source is authoritative, correct, or sufficient for the claim.
- Six-state arbitration is heuristic. Mapping `superseded` to the display label `ÇELİŞİYOR` can overstate contradiction; downstream clients should preserve the internal state.
- Optional NLI scores are not calibrated probabilities and may be biased by model training data, language, or domain. No model is bundled.
- The 63-case test is project-curated, small, and uses private fixtures. The 702-claim benchmark and reported error rates are unavailable and unverified.
- No claims are made about robustness, fairness, cross-site generalization, or comparative superiority.

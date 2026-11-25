# Architecture

## Component boundary

The package is organized as a pure Python verification layer. Callers supply claims, candidate chunks, and optionally the current chunk IDs. The package does not retrieve evidence or access YAPZETA services.

1. `__init__.py` atomizes incoming text, chooses up to three chunks by lexical overlap unless chunk IDs are explicitly provided, and returns one result per atomic claim.
2. `textutil.py` normalizes and tokenizes text, computes lexical overlap, and extracts page/year helpers.
3. `checks.py` applies deterministic checks (including counts, materials, dates, negation, terminology, and attribution patterns). Optional ONNX NLI scores and a caller-provided judge hook are auxiliary. The NLI model is not included.
4. `gate.py` annotates source metadata, checks current chunk identity, evaluates source authority/history, requires a verbatim evidence span for support, applies subject/floor checks, and maps six internal states into three user-facing labels.

## Decision path

```text
claim + supplied chunks
        ↓
atomic claim splitting and candidate selection
        ↓
deterministic checks (+ optional NLI signal)
        ↓
source / version / quote / subject checks
        ↓
six internal states → three display labels
```

The order used to select a result from multiple candidate chunks is encoded in `gate.evaluate`; an authoritative contradiction is preferred, then historical/superseded, supporting, contradictory, pending-conflict, insufficient, and finally the first remaining candidate. This arbitration policy is heuristic and requires broader testing.

## Current API

`verify_claims(claims, chunks, current_ids=None) -> list[dict]`

Chunks use the current Turkish keys `id`, `kaynak`, and `metin`, with optional `etiket` and `meta`. Returned records include the internal state, display label, reason, conflict categories, structured claim fields, and evidence metadata when supported.

## Relationship to the older `stage` implementation

The inspected `stage/verify.py` uses three outcomes (`DESTEKLİYOR`, `ÇELİŞİYOR`, `YETERSİZ`) and combines deterministic rules with optional NLI and an optional LLM judge. `stage/gate.py` already defines six internal statuses, but imports top-level `rag` and `verify` and has no three-label mapping or packaged standalone API. The `yapzeta_verifier` package is a benchmark branch: it separates `checks`, `gate`, and text helpers; provides `verify_claims`; adds status-to-label mapping, evidence offsets/paragraphs, subject/floor grounding checks, terminology checks, and history handling. Its code comment identifies it as a bench-only branch; this release does not claim it is integrated into the full YAPZETA application.

## Runtime dependencies

NumPy is required by `checks.py`. ONNX Runtime and `tokenizers` are imported only when an optional local NLI model is enabled and available. The model files are not distributed, and users must independently review model provenance and license. The optional judge hook is caller-supplied and disabled by default.

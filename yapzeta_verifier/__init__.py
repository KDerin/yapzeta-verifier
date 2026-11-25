"""yapzeta_verifier — model-bağımsız iddia doğrulayıcı (YAPZETA katman 4).

Arayüz:  verify_claims(claims, chunks, current_ids=None) -> list[dict]
  claims : list[str] | list[{"text": str, "chunk_ids": [..]?}]   (her biri tek atomik iddia ya da serbest metin; otomatik bölünür)
  chunks : list[{"id","kaynak","metin","etiket"?, "meta"?}]       (kanıt parçaları; kaynağı ne olursa olsun)
  çıktı  : her atomik iddia için {text, verification_status (6 iç durum), user_label (DESTEKLENİYOR|ÇELİŞİYOR|YETERSİZ_KANIT),
           verification_reason, conflict_categories, evidence[{doc_id, chunk_id, page, paragraph, evidence_text, verification_result}], ...}
Hiçbir dil modeline (Gemma/Qwen/yapzeta-uzman) bağımlı değildir. NLI (mDeBERTa-xnli ONNX) yalnız yardımcı sinyaldir;
kurulu değilse deterministik kurallarla çalışır. İsteğe bağlı hakem: checks.set_judge(fn) — bağlayıcı değildir.
"""
from . import textutil, checks, gate
from .gate import SUP, CON, INS, PEND, SUPD, UNV, USER_LABEL, user_label, VERSION

def verify_claims(claims, chunks, current_ids=None):
    by = {c["id"]: c for c in chunks}
    ids = set(by) if current_ids is None else set(current_ids)
    out, n = [], 0
    for cl in claims:
        text = cl if isinstance(cl, str) else cl["text"]
        cands = [by[i] for i in (cl.get("chunk_ids") or []) if i in by] if isinstance(cl, dict) and cl.get("chunk_ids") else \
                sorted(chunks, key=lambda h: -textutil.overlap(text, h.get("kaynak", "") + " " + h["metin"]))[:3]
        for a in checks.iddialar(text) or [text]:
            n += 1; out.append(gate.evaluate(a, cands, ids, f"claim-{n:05d}"))
    return out

__all__ = ["verify_claims", "user_label", "USER_LABEL", "SUP", "CON", "INS", "PEND", "SUPD", "UNV", "VERSION", "checks", "gate"]

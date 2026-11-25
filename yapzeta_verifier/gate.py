"""YAPZETA YZ — Katman 4: BAĞIMSIZ, KURAL TABANLI KANIT DENETİMİ (SARTNAME_v1 §2, §3).
LLM'den ayrı çalışır; dil modeli çağırmaz. (NLI çapraz kodlayıcı yalnız yardımcı sinyaldir; tek başına otorite değildir.)
Her atomik iddia için: kaynak/parça kimliği varlığı + güncel indeks sürümü, quote_span'in kaynakta kelimesi kelimesine bulunması,
özne/kat/sayı/tarih/darp yeri/olumsuzlama (verify.deterministik), kaynak yetkisi (source_type/approval_status), sürüm/tarihçe.
Durumlar: supported · contradicted · insufficient · conflict_pending · superseded · unverified"""
from __future__ import annotations
import hashlib, json, re
from . import textutil as rag, checks as verify

VERSION = "gate-1.1-branch (terminoloji + bölücü; yalnız bench dalı)"
SUP, CON, INS, PEND, SUPD, UNV = "supported", "contradicted", "insufficient", "conflict_pending", "superseded", "unverified"
ROZET = {SUP: "✓ destekli", CON: "✗ çelişik", INS: "? yetersiz", PEND: "⚖ çözümsüz çelişki", SUPD: "⟲ eski karar (superseded)", UNV: "⊘ doğrulanamadı"}
# Kullanıcıya gösterilen 3 etiket (iç 6 durum bunlara eşlenir)
DESTEKLENIYOR, CELISIYOR, YETERSIZ_KANIT = "DESTEKLENİYOR", "ÇELİŞİYOR", "YETERSİZ_KANIT"
USER_LABEL = {SUP: DESTEKLENIYOR, CON: CELISIYOR, SUPD: CELISIYOR, INS: YETERSIZ_KANIT, PEND: YETERSIZ_KANIT, UNV: YETERSIZ_KANIT}
def user_label(status: str) -> str: return USER_LABEL.get(status, YETERSIZ_KANIT)
GUVENILIR_DEGIL = (CON, INS, UNV, PEND, SUPD)   # kesin bilgi olarak yanıta girmez

# ---------- kaynak metadata (§3.1) ----------
def source_meta(c: dict) -> dict:
    k = c.get("kaynak", ""); m = dict(c.get("meta") or {})
    if "source_type" not in m:
        if c.get("id") == "OCRE" or k.startswith("OCRE"): st = "catalogue"
        elif c.get("etiket") == "web kaynağı" or k.startswith("web kaynağı"): st = "web"
        elif k.startswith("Saha kaydı"): st = "primary_excavation"
        elif k.startswith(("Karar notu", "Proje kaydı")): st = "project_record"
        elif k.startswith("bazilika_params"): st = "project_parameter"
        elif k.startswith(("Akademik yayın", "Literatür [JSTOR")): st = "peer_reviewed"
        elif k.startswith("Literatür"): st = "literature"
        elif k.startswith("Hocam belgesi"): st = "uploaded_document"
        elif c.get("etiket") == "doğrulanmamış not" or re.search(r"\.md\b", k): st = "chat_note"
        else: st = "other"
        m["source_type"] = st
    base = re.sub(r",?\s*(PDF s\.[^,]*|\(karakter \d+\)|\(yaklaşık dergi[^)]*\))", "", k).strip()
    m.setdefault("source_id", c.get("belge") or ("S" + hashlib.md5(base.encode()).hexdigest()[:10]))
    m.setdefault("chunk_id", c.get("id"))
    m.setdefault("title", base[:200])
    am = re.search(r"\]:\s*([^()]+?)\s*\((\d{4})\)", k)
    if am: m.setdefault("author", am.group(1).strip()); m.setdefault("publication_date", am.group(2))
    ev = re.search(r"(\d{4}-\d{2}-\d{2})", k)
    if ev: m.setdefault("event_date", ev.group(1))
    m.setdefault("page", rag.sayfa(k))
    m.setdefault("license", c.get("lisans_sinifi") or ("ODbL" if m["source_type"] == "catalogue" else None))
    m.setdefault("version", c.get("h"))
    m.setdefault("approval_status", {"project_record": "pending_primary_record", "primary_excavation": "reported", "catalogue": "published",
                                     "peer_reviewed": "published"}.get(m["source_type"], "unreviewed"))
    m.setdefault("approved_by", None); m.setdefault("approved_at", None); m.setdefault("primary_approval_record_id", None)
    oc = re.search(r"ocr[^0-9]{0,12}(0\.\d+|\d{2}(?:\.\d)?)\s*%?", (c.get("ham_etiket", "") + " " + k).lower())
    if "ocr_confidence" not in m and oc:
        v = float(oc.group(1)); m["ocr_confidence"] = v / 100 if v > 1 else v
    if m.get("ocr_confidence") is not None: m.setdefault("needs_manual_review", m["ocr_confidence"] < 0.8)
    return m

# ---------- quote_span ----------
def find_quote(claim: str, c: dict) -> str | None:
    """Kaynak metninde kelimesi kelimesine bulunan, iddiayla en çok örtüşen cümle/alan."""
    t = c.get("metin", "")
    parts = [p.strip() for p in re.split(r"(?<=[.;\n])\s+|\n|\s/\s|\s\|\s", t) if len(p.strip()) > 3]
    if not parts: return None
    best = max(parts, key=lambda p: rag.overlap(claim, rag.okunur({"kaynak": c.get("kaynak", ""), "metin": p}) + " " + p))
    return best if best in t else None

def quote_ok(chunk: dict | None, quote: str | None) -> bool:
    return bool(chunk and quote and quote in chunk.get("metin", ""))

# ---------- atomik iddia yapısı (§3.3) ----------
def structure(text: str) -> dict:
    lo = verify.low(text)
    floor = "ground" if re.search(r"zemin kat|birinci kat|alt kat", lo) else "upper" if re.search(r"üst kat|ikinci kat", lo) else None
    pred, obj = None, None
    if verify.kat_sayilari(text): pred, obj = "storey_count", sorted(verify.kat_sayilari(text))[0]
    elif any(k in lo for k in verify.MALZ): pred, obj = "material", next(v for k, v in verify.MALZ.items() if k in lo)
    elif re.search(r"darphane|mint", lo): pred = "mint"; obj = next((d for d in verify.DARPHANE if re.search(rf"\b{d}", lo)), None)
    elif verify.yillar(text) or verify.yuzyil(text): pred, obj = "date", (verify.yillar(text) or sorted(verify.yuzyil(text)))
    subj = "stoa" if "stoa" in lo else "bazilika" if "bazilika" in lo else "sikke" if re.search(r"sikke|ric |darphane", lo) else \
           "seramik" if re.search(r"seramik|kap\b|black", lo) else None
    return {"subject": subj, "predicate": pred, "object": obj, "qualifiers": {"floor": floor, "site": "Thyateira" if "thyateira" in lo else None},
            "polarity": "negative" if verify.olumsuz(text) else "positive", "time_scope": None}

def is_history(c: dict, meta: dict) -> bool:
    return (meta.get("approval_status") == "superseded" or meta.get("record_role") == "history" or
            bool(re.search(r"superseded|tarihçe", (c.get("etiket", "") + " " + c.get("kaynak", "")), re.I)))

ESKI = re.compile(r"geçersiz|superseded|tarihçe|eski|önceki|14\.10\.2026", re.I)
SON = re.compile(r"\bson\b|güncel|son karar|kesin karar", re.I)

def _tek(claim: str, c: dict, current_ids: set | None) -> dict:
    meta = source_meta(c)
    if current_ids is not None and c.get("id") not in current_ids and meta["source_type"] != "catalogue":
        return {"status": UNV, "reason": f"parça kimliği güncel indekste yok (eski sürüm/geçersiz): {c.get('id')}", "meta": meta}
    prem = verify.premise_of(c) if meta["source_type"] != "catalogue" else f"OCRE kaydı\n{c['metin']}"
    r = verify.karar(claim, prem, c.get("etiket", ""))
    st = {verify.DEST: SUP, verify.CEL: CON, verify.YET: INS}[r["karar"]]; why = r["neden"]
    # sürüm / tarihçe: iddia eski (superseded) değeri savunuyorsa
    if st == CON and ESKI.search(prem):
        old = verify.kat_sayilari(" ".join(s for s in re.split(r"(?<=[.;])\s+|\n", prem) if ESKI.search(s)))
        if verify.kat_sayilari(claim) & old:
            st = CON if SON.search(claim) or verify.HOCA_ATIF.search(claim) else SUPD
            why = f"eski kayıt (tarihçe) değeri {sorted(old)}; güncel çalışma değeri farklı — {why}"
    # tarihçe kaydı ≠ yetki: eski/superseded kayıt güncel değeri geçersiz kılamaz (ayrı kayıtlar)
    if st == CON and is_history(c, meta):
        st, why = INS, f"bulunan kayıt tarihçe/superseded; güncel değeri geçersiz kılmaz — {why}"
    # yetki (§3.2)
    if st == SUP:
        if meta["source_type"] == "chat_note": st, why = INS, "sohbet notu akademik doğrulama kaynağı sayılamaz (§3.2)"
        elif meta["source_type"] == "web": st, why = INS, "web kaynağı birincil kayıt değil; ayrıca doğrulanmalı (§3.2)"
        elif meta.get("needs_manual_review"): st, why = INS, f"OCR güveni düşük ({meta.get('ocr_confidence')}); uzman incelemesi gerekli"
        elif verify.HOCA_ATIF.search(claim) and not (meta["source_type"] == "expert_decision" and meta["approval_status"] == "approved" and meta.get("primary_approval_record_id")):
            st, why = INS, "Hoca'ya atıf: özgün onay kaydı yok (primary_approval_record_id=null); bilgi kullanıcı aktarımıdır"
    if st == SUP and c.get("celiski") and c.get("etiket") == "açık soru": st, why = PEND, "eş yetkili kaynaklar çözümsüz (açık soru)"
    out = {"status": st, "reason": why, "nli": r.get("nli"), "meta": meta, "kategoriler": r.get("celiski_kategorileri", []), "deterministik": r.get("deterministik")}
    if st == SUP:
        q = find_quote(claim, c) if meta["source_type"] != "catalogue" else next((l for l in c["metin"].splitlines() if rag.overlap(claim, l) > 0 or any(w in verify.low(l) for w in re.findall(r"\w{4,}", verify.low(claim)))), None)
        if not q or (meta["source_type"] != "catalogue" and not quote_ok(c, q)):
            out.update(status=UNV, reason="destekleyen alıntı aralığı kaynakta kelimesi kelimesine bulunamadı")
        else:
            out["quote_span"] = q
            mt = c.get("metin", ""); off = mt.find(q)
            out["offset"] = off; out["paragraph"] = mt[:off].count("\n\n") + 1 if off >= 0 else None
            # konu/kat eşleşmesi (§3.4): alıntı + kaynak başlığı iddianın öznesini ve katını anmalı
            s = structure(claim); ctx = verify.low(q + " " + c.get("kaynak", "") + " " + rag.okunur({"kaynak": c.get("kaynak", ""), "metin": q}))
            konu = {"stoa": r"stoa", "bazilika": r"bazilika|basilica", "seramik": r"seramik|kap|black"}.get(s["subject"])
            kat = {"ground": r"zemin kat|birinci kat|alt kat|ground", "upper": r"üst kat|ikinci kat|upper"}.get(s["qualifiers"]["floor"])
            if meta["source_type"] != "catalogue" and ((konu and not re.search(konu, ctx)) or (kat and not re.search(kat, ctx))):
                out.update(status=INS, reason=f"konu/kat eşleşmiyor: alıntı iddianın öznesini ({s['subject']}) veya katını anmıyor — {q[:80]}")
                out.pop("quote_span", None)
    return out

YETKILI = {"primary_excavation", "project_record", "project_parameter", "peer_reviewed", "catalogue", "expert_decision", "literature", "uploaded_document"}
def evaluate(claim: str, candidates: list[dict], current_ids: set | None = None, claim_id: str = "claim-000") -> dict:
    res = [(c, _tek(claim, c, current_ids)) for c in candidates]
    auth_con = [x for x in res if x[1]["status"] == CON and x[1]["meta"]["source_type"] in YETKILI]
    pick = (auth_con[:1] or [x for x in res if x[1]["status"] == SUPD][:1] or [x for x in res if x[1]["status"] == SUP][:1] or
            [x for x in res if x[1]["status"] == CON][:1] or [x for x in res if x[1]["status"] == PEND][:1] or
            [x for x in res if x[1]["status"] == INS][:1] or res[:1])
    j = {"claim_id": claim_id, "text": claim, **structure(claim), "evidence": [], "contradicting_evidence": [],
         "verification_status": UNV, "user_label": user_label(UNV), "rozet": ROZET[UNV], "verification_reason": "aday kaynak yok", "evidence_ids": [], "conflict_categories": []}
    if not pick:
        j.update(rozet=ROZET[UNV], user_label=user_label(UNV))
        return j
    c, r = pick[0]
    j["verification_status"], j["verification_reason"] = r["status"], r["reason"]
    j["conflict_categories"] = r.get("kategoriler", []); j["deterministic_checks"] = r.get("deterministik"); j["nli"] = r.get("nli")
    if r["status"] == SUP:
        j["evidence"] = [{"source_id": r["meta"]["source_id"], "chunk_id": c.get("id"), "quote_span": r["quote_span"], "source_type": r["meta"]["source_type"],
                          "page": r["meta"].get("page"), "paragraph": r.get("paragraph"), "char_offset": r.get("offset"), "doc_id": r["meta"]["source_id"],
                          "evidence_text": r["quote_span"], "evidence_findable": quote_ok(c, r["quote_span"]) or r["meta"]["source_type"] == "catalogue",
                          "verification_result": r["status"], "kaynak": c.get("kaynak")}]
        j["evidence_ids"] = [f"{r['meta']['source_id']}:{c.get('id')}"]
    elif r["status"] in (CON, SUPD, PEND):
        j["contradicting_evidence"] = [{"source_id": r["meta"]["source_id"], "chunk_id": c.get("id"), "source_type": r["meta"]["source_type"], "kaynak": c.get("kaynak")}]
    j["rozet"] = ROZET[j["verification_status"]]; j["user_label"] = user_label(j["verification_status"])
    return j

def validate_evidence(claim: str, evidence: list[dict], chunk_by_id: dict) -> dict:
    """Ajanın/istemcinin verdiği {chunk_id, quote_span} kanıtını denetler (T06). Geçersizse unverified."""
    for e in evidence or []:
        c = chunk_by_id.get(e.get("chunk_id"))
        if not c: return {"verification_status": UNV, "user_label": user_label(UNV), "rozet": ROZET[UNV], "verification_reason": f"chunk_id mevcut değil: {e.get('chunk_id')}", "text": claim, "evidence_ids": []}
        if not quote_ok(c, e.get("quote_span")): return {"verification_status": UNV, "user_label": user_label(UNV), "rozet": ROZET[UNV], "verification_reason": "quote_span kaynakta kelimesi kelimesine yok", "text": claim, "evidence_ids": []}
    cands = [chunk_by_id[e["chunk_id"]] for e in evidence or []]
    if not cands: return {"verification_status": UNV, "user_label": user_label(UNV), "rozet": ROZET[UNV], "verification_reason": "kanıt verilmedi", "text": claim, "evidence_ids": []}
    j = evaluate(claim, cands, set(chunk_by_id))
    if j["verification_status"] == SUP:   # verilen alıntı da iddiayı taşımalı
        q = {e["chunk_id"]: e["quote_span"] for e in evidence}
        j["evidence"][0]["quote_span_given"] = q.get(j["evidence"][0]["chunk_id"])
    return j

# ---------- ajan çıktısına uygula ----------
_seq = [0]
def apply_agent(a: dict, hits: list[dict], coin: dict | None, current_ids: set | None) -> dict:
    ocre = None
    if coin and coin.get("yerel_kayit"):
        uri = next((s["kaynak"] for i in coin.get("ifadeler", []) for s in i.get("kaynaklar", []) if "ocre" in s.get("kaynak", "")), "OCRE")
        ocre = {"id": "OCRE", "kaynak": f"OCRE kaydı (ODbL) {uri}", "etiket": "OCRE kaydı", "meta": {"source_type": "catalogue", "source_id": uri, "license": "ODbL", "approval_status": "published"},
                "metin": "\n".join(i["metin"] for i in coin.get("ifadeler", []) if i.get("etiket", "").startswith("OCRE")).replace("Darphane:", "Mint:")}
    a["claims"] = []; a["dogrulama_celiski"] = []
    for it in a["ifadeler"]:
        if it["durum"] in ("bilgi yok", "hata"): continue
        cited = [hits[s["no"] - 1] for s in it["kaynaklar"] if s.get("no") and 1 <= s["no"] <= len(hits)]
        oto = it["durum"] == "otomatik atıf" or not cited
        cands = cited if not oto else sorted(hits, key=lambda h: -rag.overlap(it["metin"], h["kaynak"] + " " + h["metin"]))[:3]
        if ocre and re.search(r"darphane|mint|tarih|ms\s*\d|mö\s*\d|basıl|darp", verify.low(it["metin"])): cands = [ocre] + cands
        cl = []
        for c in verify.iddialar(it["metin"]):
            _seq[0] += 1
            cl.append(evaluate(c, cands, current_ids, f"claim-{_seq[0]:05d}"))
        if not cl: continue
        order = [CON, SUPD, PEND, UNV, INS, SUP]
        worst = min((x["verification_status"] for x in cl), key=order.index)
        it["dogrulama"] = {"karar": worst, "rozet": ROZET[worst], "etiket": user_label(worst), "iddialar": [{"iddia": x["text"], "rozet": x["rozet"], "etiket": x["user_label"], "neden": x["verification_reason"], "kategoriler": x["conflict_categories"],
                           "kaynak": (x["evidence"] or x["contradicting_evidence"] or [{}])[0].get("kaynak")} for x in cl]}
        if worst == SUP:
            ev = cl[0]["evidence"][0]
            n = next((i for i, h in enumerate(hits, 1) if h["id"] == ev["chunk_id"]), None)
            it["kaynaklar"] = [{"no": n, "id": ev["chunk_id"], "etiket": next((h["etiket"] for h in hits if h["id"] == ev["chunk_id"]), "OCRE kaydı"),
                                "kaynak": ev["kaynak"], "sayfa": ev.get("page"), "quote_span": ev["quote_span"]}]
            it["durum"] = "doğrulandı"
        else:
            it["kaynaklar"] = []   # §3.5: doğrulanmamış iddiaya kaynak etiketi eklenmez
            it["durum"] = {CON: "çelişik", SUPD: "eski karar", PEND: "çözümsüz çelişki", UNV: "doğrulanamadı", INS: "kaynak yetersiz"}[worst]
            it["etiket"] = "Açık çelişki / doğrulanamayan iddia"
            if worst in (CON, SUPD):
                bad = next(x for x in cl if x["verification_status"] == worst)
                ce = (bad["contradicting_evidence"] or [{}])[0]
                a["dogrulama_celiski"].append({"konu": f"denetim — {a['ajan']} iddiası: {ROZET[worst]}", "kaynak_turu": "kanıt denetimi (katman 4)",
                    "taraflar": [{"ajan": a["ajan"], "metin": bad["text"], "etiket": "ajan iddiası", "kaynaklar": []},
                                 {"ajan": "kaynak", "metin": bad["verification_reason"], "etiket": ce.get("source_type", ""),
                                  "kaynaklar": [{"no": None, "id": ce.get("chunk_id"), "etiket": ce.get("source_type", ""), "kaynak": ce.get("kaynak", "")}]}]})
        a["claims"] += [dict(x, ajan=a["ajan"]) for x in cl]
    return a

"""Model-bağımsız metin yardımcıları (rag.py / orchestrator.py'den kopya; LLM, gömme ya da indeks bağımlılığı yok)."""
from __future__ import annotations
import json, re

def tok(t: str) -> list[str]:
    t = t.lower().replace("i̇", "i").replace("ı", "i")
    return [w[:5] for w in re.findall(r"\w+", t) if len(w) > 2]

STOP = set(tok("için ile olan olarak bir bu ve veya the and of to in is are not değil daha gibi kadar"))

def overlap(stmt: str, src: str) -> float:
    a = set(tok(re.sub(r"\[\d+\]", "", stmt))) - STOP
    if not a: return 1.0
    b = set(tok(src))
    return len(a & b) / len(a)

KAT_ADLARI = [(r"(?i)birinci kat(ın)? tabanı|level 1 floor surface|floors\.0\.surface", "Zemin kat (birinci kat) zemin kaplaması"),
              (r"(?i)ikinci kat(ın)? tabanı|level 2 floor surface|floors\.1\.surface", "Üst kat (ikinci kat) zemin kaplaması")]

def okunur(h: dict) -> str:
    m = h["metin"]
    try:
        d = json.loads(m)
        if isinstance(d, dict):
            if "label_tr" in d and "value" in d: m = f"{d['label_tr']}: {d['value']} (durum: {d.get('status', '-')})"
            elif "tr" in d: m = d["tr"]
    except Exception:
        pass
    if h.get("kaynak", "").startswith(("Saha kaydı", "bazilika_params")) or "floors." in h.get("kaynak", ""):
        m = re.sub(r"\s/\s[A-Z][^.\n]*?(?=:|\.|$)", "", m)
        for pat, rep in KAT_ADLARI: m = re.sub(pat, rep, m)
        if "floors.0.surface" in h.get("kaynak", "") and "Zemin kat" not in m.split(":")[0]:
            m = "Zemin kat (birinci kat) zemin kaplaması: " + m
    return m

ROMAN = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6, 'vii': 7, 'viii': 8, 'ix': 9, 'x': 10, 'xi': 11, 'xii': 12, 'xiii': 13, 'xiv': 14, 'xv': 15}
def yuzyillar(t: str) -> set[int]:
    s, low = set(), t.lower()
    for era, n in re.findall(r"(m[öo]|ms|bc|ad)?\s*(\d{1,2}|[ivx]{1,5})\.?\s*(?:yüzyıl|yy|century|cent)", low):
        v = int(n) if n.isdigit() else ROMAN.get(n)
        if v: s.add(-v if era in ("mö", "mo", "bc") else v)
    for era, y in re.findall(r"\b(m[öo]|ms|bc|ad)\s*(\d{2,4})\b", low):
        c = (int(y) - 1) // 100 + 1; s.add(-c if era in ("mö", "mo", "bc") else c)
    return s

PAGE_RE = re.compile(r"(PDF s\. ?[\d\-–]+(?: \(yaklaşık dergi s\. ?[^)]*\))?|ss\. ?[\d\-–]+|karakter \d+|s\. ?\d+[\-–]?\d*)")
def sayfa(kaynak: str) -> str | None:
    m = PAGE_RE.findall(kaynak or ""); return m[-1] if m else None

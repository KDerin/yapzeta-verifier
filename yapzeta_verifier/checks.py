"""YAPZETA YZ - akademik kanıt doğrulama katmanı (Aşama 2).
Her ajan ifadesi atomik iddialara bölünür; her iddia + atıf yapılan parça(lar) için karar:
  DESTEKLİYOR / ÇELİŞİYOR / YETERSİZ
Yöntem: (1) deterministik kontroller — kat sayısı, zemin/üst kat malzemesi, tarih/yüzyıl, darphane, sayı, olumsuzluk,
'Hocam ... dedi' türü atıf (Hoca kaynaklı parça gerekir; 'geçersiz/tarihçe' cümleleri kabul edilen değer sayılmaz);
(2) yerel çok dilli NLI çapraz kodlayıcı (mDeBERTa-v3-base-xnli, ONNX int8, CPU; boşta 10 dk sonra bellekten atılır);
(3) isteğe bağlı, bağlayıcı olmayan hakem kancası (set_judge); model-bağımsız.
'otomatik atıf' yalnızca DESTEKLİYOR ise kalır."""
from __future__ import annotations
import json, os, re, threading, time
from pathlib import Path
import numpy as np
from . import textutil as rag

NLI_DIR = Path(os.environ.get("YZ_NLI_DIR", ""))
NLI_IDLE = int(os.environ.get("YZ_NLI_IDLE_SN", "600"))
USE_NLI = os.environ.get("YZ_VERIFY_NLI", "0") == "1"
USE_LLM = os.environ.get("YZ_VERIFY_LLM", "0") == "1"
DEST, CEL, YET = "DESTEKLİYOR", "ÇELİŞİYOR", "YETERSİZ"
ROZET = {DEST: "✓ destekli", CEL: "✗ çelişik", YET: "? yetersiz"}

# ---------------- NLI (ONNX) ----------------
class _NLI:
    def __init__(self):
        self.sess = self.tok = None; self.lock = threading.Lock(); self.last = 0; self.err = None
        self.lab = {}
    def ok(self):
        return USE_NLI and (NLI_DIR / "model_quantized.onnx").exists() and (NLI_DIR / "tokenizer.json").exists()
    def _load(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        so = ort.SessionOptions(); so.intra_op_num_threads = int(os.environ.get("YZ_NLI_THREADS", "4"))
        self.sess = ort.InferenceSession(str(NLI_DIR / "model_quantized.onnx"), so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(NLI_DIR / "tokenizer.json"))
        self.tok.enable_truncation(max_length=384); self.tok.enable_padding()
        cfg = json.loads((NLI_DIR / "config.json").read_text(encoding="utf-8"))
        self.lab = {int(k): v.lower() for k, v in cfg["id2label"].items()}
        self.inputs = {i.name for i in self.sess.get_inputs()}
        threading.Thread(target=self._idle, daemon=True).start()
    def _idle(self):
        while self.sess is not None:
            time.sleep(30)
            if time.time() - self.last > NLI_IDLE:
                with self.lock: self.sess = self.tok = None   # RAM'i bırak
    def scores(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """[(öncül, hipotez)] -> [{'entailment','neutral','contradiction'}]"""
        if not pairs or not self.ok(): return []
        with self.lock:
            try:
                if self.sess is None: self._load()
            except Exception as e:
                self.err = str(e); return []
            self.last = time.time(); out = []
            for i in range(0, len(pairs), 8):
                enc = self.tok.encode_batch(pairs[i:i + 8])
                feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                        "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
                if "token_type_ids" in self.inputs: feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
                lg = self.sess.run(None, feed)[0]
                p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
                out += [{self.lab[j]: float(r[j]) for j in range(len(r))} for r in p]
            return out
NLI = _NLI()

# ---------------- yardımcılar ----------------
def low(t): return (t or "").replace("İ", "i").replace("I", "ı").lower()

def iddialar(metin: str) -> list[str]:
    """İfadeyi atomik iddialara böl (cümle / ';' / ' | ')."""
    t = re.sub(r"\[\d+[^\]]*\]", "", metin or "")
    t = re.sub(r"\b(M[ÖSO]|m[öso]|BC|AD)\s*(\d{1,2})\.(?=\s)", r"\1 \2§", t)
    t = re.sub(r"\b(\d{1,2})\.(?=\s+(yüzyıl|yy\b|yüzyılın|binyıl|çeyre|yarı))", r"\1§", t, flags=re.I)   # 'sonu 5. yüzyıl' bölünmesin        # 'MÖ 6. yüzyıl' cümleyi bölmesin
    parts = [p.replace("§", ".") for p in re.split(r"(?<=[.!?;])\s+|\s\|\s|;\s*", t)]
    out = [p.strip(" -•:") for p in parts if len(re.findall(r"\w+", p)) >= 3
           and not p.strip().endswith("?") and not re.match(r"\s*(soru|question)\s*:", p, re.I)]   # soru yankısı iddia değildir
    return out or ([t.strip()] if t.strip() else [])

SAYI = {"tek": 1, "bir": 1, "iki": 2, "üç": 3, "dört": 4}
def kat_sayilari(t: str) -> set[int]:
    s = set()
    for n, _ in re.findall(r"\b(\d+|tek|iki|üç|dört)\s*(katlı|kat\b|katlıdır|storey)", low(t)): s.add(SAYI.get(n) or int(n))
    for n in re.findall(r"stoastoreys\s*[=:]\s*(\d)", low(t)): s.add(int(n))
    for n in re.findall(r"stoa kat sayısı[^:]*:\s*(\d)", low(t)): s.add(int(n))
    return s

GECERSIZ = re.compile(r"geçersiz|tarihçe|eski |önceki|superseded", re.I)
def kabul_edilen(premise: str) -> str:
    """Geçersiz/tarihçe cümlelerini çıkar -> kabul edilen bilgi."""
    return " ".join(c for c in re.split(r"(?<=[.;])\s+|\n", premise) if not GECERSIZ.search(c))

MALZ = {"opus sectile": "opus sectile", "opus sectila": "opus sectile", "tuğla": "tuğla", "mermer": "mermer", "mozaik": "mozaik", "ahşap": "ahşap", "toprak": "toprak"}
KAT_RE = re.compile(r"(zemin kat|birinci kat|alt kat|level 1|floors\.0|üst kat|ikinci kat|level 2|floors\.1)")
def kat_malzeme(t: str) -> dict:
    t = low(t); out = {}
    ms = list(KAT_RE.finditer(t))
    for i, m in enumerate(ms):
        seg = t[m.start(): ms[i + 1].start() if i + 1 < len(ms) else len(t)]
        if i + 1 < len(ms) and ms[i + 1].start() - m.end() < 4: continue   # 'Zemin kat (birinci kat)' -> sonraki ile aynı
        kat = "Z" if m.group(1) in ("zemin kat", "birinci kat", "alt kat", "level 1", "floors.0") else "U"
        for k, v in MALZ.items():
            if k in seg: out.setdefault(kat, set()).add(v)
    return out

def yillar(t: str) -> list[int]:
    out = []
    for era, a, b in re.findall(r"\b(m[öo]|ms)\s*(\d{1,4})(?:\s*[–\-]\s*(?:m[öo]|ms)?\s*(\d{1,4}))?", low(t)):
        for y in (a, b):
            if y: out.append(-int(y) if era.startswith("m") and era != "ms" else int(y))
    return out

def yuzyil(t: str) -> set[int]:
    return rag.yuzyillar(t)

DARPHANE = ["efes", "ephesos", "roma", "rome", "thyateira", "sardis", "smyrna", "izmir", "pergamon", "antiochia", "antakya", "alexandria",
            "iskenderiye", "lugdunum", "nicomedia", "kyzikos", "philadelpheia", "hierapolis", "milano", "siscia", "ticinum", "merve"]
def darphane_premise(p: str):
    m = re.search(r"(?:mint|darphane)\s*:\s*([^\n,;]+)", p, re.I)
    return low(m.group(1)).strip() if m else None

NEG = re.compile(r"\b(değil\w*|yok\w*|hiçbir|asla)\b|\w+(m[ae]d[ıi]|m[ae]z|m[ae]m[ıi]ş|m[ae]makta|m[ae]mıştır)\b")
def olumsuz(t): return bool(NEG.search(low(t)))

NEG_SUF = re.compile(r"^(\w+?)(m[ae]z|m[ae]d[ıi]|m[ae]m[ıi]ş(t[ıi]r)?|m[ae]makta(d[ıi]r)?|m[ae]y[ae]n|m[ae]m[ae]l[ıi]|m[ıiuü]yor(du|lar|sa)?)$")
def neg_kokler(t: str) -> set[str]:
    """Olumsuz kökler: 'üretilmez' -> 'üreti'; 'net değil' -> 'net'; 'apsis yoktur' -> 'apsis'."""
    w = re.findall(r"\w+", low(t)); out = set()
    for i, x in enumerate(w):
        m = NEG_SUF.match(x)
        if m and len(m.group(1)) >= 4: out.add(m.group(1))   # tam kök: "gösteril" ≠ "gösteriyor"
        elif re.fullmatch(r"değil\w*|yok\w*", x) and i: out.add(w[i - 1][:5])
    return out

HOCA_ATIF = re.compile(r"(hoca|hocam|uzman)\b[^.]{0,80}\b(dedi|söyledi|belirtti|bildirdi|karar|teyit|onayla)", re.I)

VARLIK = re.compile(r"\b(vardır|var\b|yaygın\w*|görülür\w*|bulunur\w*|bulunmaktadır|mevcut\w*|kullanıl\w*|bilinir|görülmektedir|tespit edil\w*)")
TERIM_AILELERI = [  # aynı ailede birbirinin yerine geçemeyen teknik terimler
    ["black-figure", "siyah figür", "black-on-red", "siyah üzeri kırmızı", "red-figure", "kırmızı figür", "bichrome", "marbled", "mermerleme", "streaky"],
    ["opus sectile", "opus tessellatum", "opus vermiculatum", "opus signinum", "mozaik"],
    ["amphora", "amfora", "krater", "kylix", "kyliks", "lekythos", "oinokhoe", "hydria", "skyphos", "pyxis", "lydion", "kantharos"],
    ["terra sigillata", "kırmızı astarlı", "sırlı", "sgraffito", "gri mal"],
    ["dor", "ion", "korinth", "toskana", "kompozit"],
    ["mermer", "kireçtaşı", "granit", "bazalt", "andezit", "tuğla", "moloz taş", "traverten"],
    ["denarius", "sestertius", "antoninianus", "aureus", "cistophorus", "tetradrahmi", "didrahmi", "drahmi", "follis", "solidus"],
    ["apsis", "nartheks", "atrium", "nef", "transept", "bema"],
    ["sütun", "pilaster", "payanda", "paye"], ["lahit", "osteotek", "urne"], ["tiyatro", "stadion", "odeion", "bouleuterion"]]
ALAN_ADLARI = ["bazilika", "stoa", "tapınak", "kilise", "agora", "seramik", "sikke", "mühür", "lahit", "sunak", "mozaik", "heykel", "nekropol", "tiyatro",
               "hamam", "yazıt", "kabartma", "kandil", "unguentarium", "osteotek", "stadion", "atrium", "nartheks", "pilaster", "fresk", "krater", "amphora", "amfora"]
KESIN = re.compile(r"\bkesin\w*|kanıtlanm\w*|tartışmasız|şüphesiz|\bmutlaka\b")
TERIM_ES = {"amfora": "amphora", "kyliks": "kylix", "siyah figür": "black-figure", "siyah üzeri kırmızı": "black-on-red", "kırmızı figür": "red-figure", "mermerleme": "marbled"}
def _terimler(s):
    lo = low(s); out = {}
    for i, fam in enumerate(TERIM_AILELERI):
        for term in fam:
            tail = r"(?!\w)" if len(term) <= 5 else (r"(?!leme)" if term == "mermer" else "")
            if re.search(rf"(?<!\w){re.escape(term)}{tail}", lo): out.setdefault(i, set()).add(TERIM_ES.get(term, term))
    return out
KISILER = {"hadrian": "Hadrianus", "traian": "Traianus", "trajan": "Traianus", "decius": "Decius", "augustus": "Augustus",
           "vespasian": "Vespasianus", "antoninus": "Antoninus Pius", "marcus aurel": "Marcus Aurelius", "septimius": "Septimius Severus",
           "caracalla": "Caracalla", "gordian": "Gordianus", "constantin": "Constantinus", "nero": "Nero", "tiberius": "Tiberius",
           "claudius": "Claudius", "domitian": "Domitianus", "commodus": "Commodus", "valerian": "Valerianus", "gallienus": "Gallienus"}
YERLER = {"akhisar": "Akhisar", "bergama": "Bergama", "manisa": "Manisa", "izmir": "İzmir", "salihli": "Salihli", "soma": "Soma",
          "kırkağaç": "Kırkağaç", "gördes": "Gördes", "turgutlu": "Turgutlu", "alaşehir": "Alaşehir", "aydın": "Aydın", "selçuk": "Selçuk", "denizli": "Denizli"}
KATEGORI = [("kat sayısı", "sayı"), ("ölçü", "sayı"), ("sayı", "sayı"), ("tarih", "tarih"), ("yüzyıl", "tarih"), ("kişi", "kişi"), ("yer", "yer"),
            ("darphane", "yer"), ("terminoloji", "terminoloji"), ("terim kaynakta", "terminoloji"), ("olumsuzluk", "olumsuzluk"), ("zemin kat", "kat/malzeme"), ("üst kat", "kat/malzeme"), ("atıf", "yetki")]
def kategoriler(nedenler: list[str]) -> list[str]:
    return sorted({next((c for p, c in KATEGORI if n.startswith(p)), "diğer") for n in nedenler})

# ---------------- tek iddia ----------------
def deterministik(claim: str, premise: str, etiket: str) -> tuple[list[str], list[str], list[str]]:
    """-> (çelişki nedenleri, eksik nedenleri, destek sinyalleri)"""
    cel, eks, dst = [], [], []
    tarihce_iddia = bool(GECERSIZ.search(claim))
    pk = premise if tarihce_iddia else kabul_edilen(premise)
    # who-said-what
    if HOCA_ATIF.search(claim):
        if etiket != "Hoca kararı" and not re.search(r"hoca|derya", low(premise)): eks.append("atıf: kaynak Hoca'ya ait değil")
        else: dst.append("Hoca kaynaklı parça")
    # [dal 1.1] dayanak (grounding) kuralları — kısmi deterministik eşleşme tek başına destek sayılmaz
    lc, lp = low(claim), low(premise)
    cn_ = set(re.findall(r"(?<![\d.,])\d+(?:[.,]\d+)?", lc)); pn_ = set(re.findall(r"(?<![\d.,])\d+(?:[.,]\d+)?", lp))
    norm = lambda xs: {x.replace(",", ".") for x in xs}
    miss = sorted(norm(cn_) - norm(pn_))
    if miss: eks.append(f"sayı kaynakta yok: {', '.join(miss[:3])}")
    # sayı dizisi: iddia ile en yakın kaynak cümlesinde aynı sayıda sayı varsa sıra/değer birebir aynı olmalı (yer değiştirme/kopya değer)
    seq = lambda s: [x.replace(",", ".") for x in re.findall(r"(?<![\d.,])\d+(?:[.,]\d+)?", low(s))]
    cs_ = seq(claim)
    if len(cs_) >= 2:
        best = max(re.split(r"(?<=[.;\n])\s+", premise) or [premise], key=lambda s: rag.overlap(claim, s))
        ps_ = seq(best)
        if len(ps_) == len(cs_) and ps_ != cs_ and rag.overlap(claim, best) >= .6: cel.append(f"sayı: değer dizisi farklı {cs_} ≠ kaynak {ps_}")
    words = re.findall(r"\w+", claim)
    props = {w for i, w in enumerate(words) if i > 0 and w[:1].isupper() and len(w) > 3 and not w.isupper()}
    pmiss = sorted(p for p in props if low(p)[:5] not in lp)
    if pmiss: eks.append(f"özel ad kaynakta yok: {', '.join(pmiss[:3])}")
    lp_ = lp + (" sikke" if lp.lstrip().startswith("ocre kaydı") else "")   # katalog kaydı örtük olarak sikkedir
    dmiss = sorted(n for n in ALAN_ADLARI if re.search(rf"(?<!\w){n}", lc) and not re.search(rf"(?<!\w){n}", lp_))
    if dmiss: eks.append(f"alan terimi/özne kaynakta yok: {', '.join(dmiss[:3])}")
    if KESIN.search(lc) and not KESIN.search(lp): eks.append("kesinlik abartısı: iddia kesin, kaynak kesinlik bildirmiyor")
    # konu (özne): iddia stoa/bazilika hakkındaysa kaynak da o yapıyı anmalı ("iki katlı bina" ≠ stoa)
    for konu, rx in (("stoa", r"stoa"), ("bazilika", r"bazilika|basilica")):
        if re.search(rx, low(claim)) and not re.search(rx, low(premise)): eks.append(f"konu: kaynak '{konu}' yapısını anmıyor")
    # alıntılanan ifade ("...") kaynakta kelimesi kelimesine geçmeli
    for qt in re.findall(r"[\"“”]([^\"“”]{8,})[\"“”]", claim):
        if len(qt.split()) >= 3 and low(qt).strip() not in low(premise): eks.append(f"alıntı: '{qt[:40]}' kaynakta yok")
    # kat sayısı
    ck, pk_k = kat_sayilari(claim), kat_sayilari(pk)
    if ck and pk_k:
        if ck & pk_k: dst.append(f"kat sayısı {sorted(ck)}")
        else: cel.append(f"kat sayısı: iddia {sorted(ck)} ≠ kaynak {sorted(pk_k)}")
    elif ck and "stoa" in low(claim): eks.append("kaynakta kat sayısı yok")
    # zemin / üst kat malzemesi
    cm, pm = kat_malzeme(claim), kat_malzeme(pk)
    for k, v in cm.items():
        if k in pm:
            if v & pm[k]: dst.append(f"{'zemin' if k == 'Z' else 'üst'} kat malzemesi")
            else: cel.append(f"{'zemin' if k == 'Z' else 'üst'} kat malzemesi: iddia {sorted(v)} ≠ kaynak {sorted(pm[k])}")
        else: eks.append("kaynakta o katın malzemesi yok")
    # darphane
    named = [d for d in DARPHANE if re.search(rf"\b{d}", low(claim))]
    if re.search(r"darphane|mint", low(claim)) or (named and re.search(r"basıl|darp", low(claim))):
        pv = darphane_premise(premise)
        if pv:
            if pv.split()[0] in low(claim): dst.append(f"darphane {pv}")
            elif named: cel.append(f"darphane: iddia {named} ≠ kaynak {pv}")
            else: eks.append("darphane adı eşleşmedi")
        elif named and not any(re.search(rf"\b{d}", low(premise)) for d in named): eks.append("kaynakta darphane yok")
    # tarih
    cy, py = yillar(claim), yillar(pk)
    if cy and py:
        lo, hi = min(py), max(py)
        if all(lo - 1 <= y <= hi + 1 for y in cy): dst.append("yıl aralığı")
        else: cel.append(f"tarih: iddia {cy} ∉ kaynak {lo}–{hi}")
    else:
        cc, pc = yuzyil(claim), yuzyil(pk) | {((y - 1) // 100 + 1 if y > 0 else -((-y - 1) // 100 + 1)) for y in py}
        if cc and pc:
            if cc & pc: dst.append("yüzyıl")
            else: cel.append(f"yüzyıl: iddia {sorted(cc)} ≠ kaynak {sorted(pc)}")
        elif (cy or cc): eks.append("kaynakta tarih yok")
    # ölçüler (m/cm)
    pnums = {x.replace(",", ".") for x in re.findall(r"(?<![\d.])\d+(?:[.,]\d+)?(?![\d])", low(pk))}
    for n, u in re.findall(r"(\d+(?:[.,]\d+)?)\s*(m|cm)\b", low(claim)):
        nn = re.escape(n).replace("\\.", "[.,]").replace(",", "[.,]")
        if re.search(rf"(?<![\d.,]){nn}\s*{u}\b", low(pk)) or (re.search(rf"\({u}\)", low(pk)) and n.replace(",", ".") in pnums):
            dst.append(f"ölçü {n} {u}")
        elif rag.overlap(claim, pk) >= .5 and re.search(rf"\d\s*{u}\b|\({u}\)", low(pk)): cel.append(f"ölçü: iddia {n} {u} kaynakta farklı ({', '.join(sorted(pnums)[:4])})")
        else: eks.append(f"ölçü {n} {u} kaynakta yok")
    # terminoloji: iddiadaki teknik terim, kaynakta aynı ailenin başka bir terimiyle değiştirilmişse çelişki; hiç geçmiyorsa eksik
    ct_, pt_ = _terimler(claim), _terimler(pk)
    for fam, terms in ct_.items():
        if fam in pt_:
            if terms & pt_[fam]: dst.append(f"terim {sorted(terms & pt_[fam])[0]}")
            elif not (pt_[fam] & terms): cel.append(f"terminoloji: iddia {sorted(terms)} ≠ kaynak {sorted(pt_[fam])}")
    # kişi (imparator/otorite, yazar)
    cp = {v for k, v in KISILER.items() if re.search(rf"\b{k}", low(claim))}
    if cp:
        pp = {v for k, v in KISILER.items() if re.search(rf"\b{k}", low(pk))}
        if pp:
            if cp & pp: dst.append(f"kişi {sorted(cp & pp)[0]}")
            elif re.search(r"otorite|imparator|authority|adına|döneminde|yazar", low(claim + " " + pk)): cel.append(f"kişi: iddia {sorted(cp)} ≠ kaynak {sorted(pp)}")
            else: eks.append("kişi eşleşmedi")
        else: eks.append("kaynakta kişi yok")
    # yer (modern il/ilçe)
    cy_ = {v for k, v in YERLER.items() if re.search(rf"\b{k}", low(claim))}
    if cy_ and re.search(r"ilçe|ilinde|\bil\b|bugün|yer al|bulun|konum", low(claim)):
        py_ = set()
        for sent in re.split(r"(?<=[.;\n])\s+", pk):
            if rag.overlap(claim, sent) >= .3: py_ |= {v for k, v in YERLER.items() if re.search(rf"\b{k}", low(sent))}
        if py_:
            if cy_ <= py_: dst.append(f"yer {sorted(cy_)[0]}")
            else: cel.append(f"yer: iddia {sorted(cy_ - py_)} ≠ kaynak {sorted(py_)}")
        else: eks.append("kaynakta yer adı yok")
    # olumsuzluk: aynı fiil/sıfat kökü kaynakta olumsuz, iddiada olumlu (ya da tersi)
    if not cel:   # olumsuz kök, iddiayla aynı konudaki cümlede olmalı (cümle örtüşmesi >= .5)
        cn = neg_kokler(claim); ct = set(re.findall(r"\w+", low(claim)))
        for sent in re.split(r"(?<=[.;\n])\s+", pk):
            if rag.overlap(claim, sent) < .3: continue
            pn, st = neg_kokler(sent), set(re.findall(r"\w+", low(sent)))
            hit = [s for s in pn if any(w.startswith(s) for w in ct) and s not in cn] + [s for s in cn if any(w.startswith(s) for w in st) and s not in pn]
            if hit: cel.append(f"olumsuzluk: '{hit[0]}…' kaynak ve iddiada zıt kutupta"); break
        # varlık olumsuzlaması: iddia 'görülmez/yoktur' der, aynı konudaki kaynak cümlesi olumlu varlık bildirir
        if not cel and olumsuz(claim):
            core = re.sub(r"\w+(m[ae]z|m[ae]d[ıi]|m[ae]m[ıi]ş\w*)\b|\b(değil|yok)\w*", " ", low(claim))
            ss = re.split(r"(?<=[.;\n])\s+", pk)
            for sent in [" ".join(ss[i:i + 2]) for i in range(len(ss))]:   # 'MÖ 7.' gibi kısaltmalar cümleyi böler: ikili pencere
                if not olumsuz(sent) and rag.overlap(core, sent) >= .6 and VARLIK.search(low(sent)):
                    cel.append("olumsuzluk: iddia varlığı reddediyor, kaynak olumlu varlık bildiriyor"); break
    return cel, eks, dst

def _windows(p: str, n=420):
    sents = re.split(r"(?<=[.;\n])\s+", p); out, cur = [], ""
    for s in sents:
        if len(cur) + len(s) > n and cur: out.append(cur); cur = ""
        cur += " " + s
    if cur.strip(): out.append(cur)
    return out[:4] or [p[:n]]

JUDGE = None   # isteğe bağlı danışman: fn(claim, premise) -> ("DESTEKLİYOR"|"ÇELİŞİYOR"|"YETERSİZ", neden). Model-bağımsız kanca.
def set_judge(fn):
    global JUDGE; JUDGE = fn
def llm_hakem(claim, premise):
    if JUDGE is None: return None, "hakem yok"
    try: return JUDGE(claim, premise)
    except Exception as e: return None, str(e)

def karar(claim: str, premise: str, etiket: str = "") -> dict:
    cel, eks, dst = deterministik(claim, premise, etiket)
    ov = rag.overlap(claim, premise)
    nli = {}
    sc = NLI.scores([(w, claim) for w in _windows(premise)]) if NLI.ok() else []
    if sc:
        nli = {"entailment": max(s["entailment"] for s in sc), "contradiction": max(s["contradiction"] for s in sc)}
    ent, con = nli.get("entailment", 0), nli.get("contradiction", 0)
    # Kural: deterministik uyuşmazlık NLI'yı her zaman geçersiz kılar; NLI tek başına karar vermez; kelime örtüşmesi tek başına yetmez.
    if cel: k, neden = CEL, "; ".join(cel)
    elif any(e.startswith("atıf") for e in eks): k, neden = YET, "; ".join(eks)
    elif not eks and len(claim.split()) >= 6 and re.sub(r"\W+", " ", low(claim)).strip() in re.sub(r"\W+", " ", low(premise)):
        k, neden = DEST, "birebir alıntı: iddia kaynakta kelimesi kelimesine geçiyor" + (f" + {', '.join(dst)}" if dst else "")
    elif sc and ent >= .6 and con < .3 and not eks and (dst or ov >= .6): k, neden = DEST, f"NLI içerim {ent:.2f} + " + (", ".join(dst) if dst else f"sözcük teyidi {ov:.2f}")
    elif dst and not eks and con < .8 and (ov >= .5 or (ov >= .25 and any(s.startswith(("darphane", "yıl aralığı", "kat sayısı", "zemin kat", "üst kat", "kişi", "yer")) for s in dst))): k, neden = DEST, "deterministik: " + ", ".join(dst) + (f" (NLI {ent:.2f})" if sc else "")
    elif sc and con >= .8 and ov >= .5: k, neden = YET, f"NLI çelişki sinyali {con:.2f}; deterministik teyit yok (NLI tek başına karar vermez)"
    else: k, neden = YET, "; ".join(eks) or (f"NLI içerim {ent:.2f}; deterministik/sözcük teyidi yok" if sc else "NLI yok; deterministik destek yok")
    if USE_LLM and k == YET and not cel and not eks and sc and .3 <= ent < .6:
        lk, ln = llm_hakem(claim, premise)
        if lk == DEST and dst: k, neden = DEST, f"hakem + deterministik: {ln} (NLI {ent:.2f})"
        elif lk: neden += f" | hakem görüşü (bağlayıcı değil): {lk} {ln}"
    return {"karar": k, "rozet": ROZET[k], "neden": neden, "nli": {a: round(b, 3) for a, b in nli.items()}, "ortusme": round(ov, 2), "celiski_kategorileri": kategoriler(cel), "deterministik": {"celiski": cel, "eksik": eks, "destek": dst}}

def premise_of(h: dict) -> str:
    return f"{h.get('kaynak', '')}\n{rag.okunur(h)}"

# ---------------- ajan çıktısına uygula ----------------
def dogrula_ajan(a: dict, hits: list[dict], coin: dict | None = None) -> dict:
    """Her ifade: iddialara böl, atıflı parça(lar)a karşı karar ver; ifade kararı = en kötü iddia (ÇELİŞİYOR > YETERSİZ > DESTEKLİYOR)."""
    ocre = None
    if coin and coin.get("yerel_kayit"):
        ocre = {"kaynak": "OCRE kaydı (ODbL)", "etiket": "OCRE kaydı", "metin": "\n".join(i["metin"] for i in coin.get("ifadeler", []) if i.get("etiket", "").startswith("OCRE")).replace("Darphane:", "Mint:"), "id": "OCRE"}
    yeni_cel = []
    for it in a["ifadeler"]:
        if it["durum"] in ("bilgi yok", "hata"): continue
        cited = [hits[s["no"] - 1] for s in it["kaynaklar"] if s.get("no") and 1 <= s["no"] <= len(hits) and s.get("id") != None]
        oto = it["durum"] == "otomatik atıf" or not cited
        if not cited:   # atıfsız: en yakın 2 parça aday, ancak DESTEKLİYOR ise bağla
            cited = sorted(hits, key=lambda h: -rag.overlap(it["metin"], h["kaynak"] + " " + h["metin"]))[:2]
        premises = [(h, premise_of(h)) for h in cited]
        if ocre and re.search(r"darphane|mint|tarih|ms\s*\d|mö\s*\d|basıl", low(it["metin"])): premises.insert(0, (ocre, f"OCRE kaydı\n{ocre['metin']}"))
        sonuc = []
        for c in iddialar(it["metin"]):
            per = [(h, karar(c, p, h.get("etiket", ""))) for h, p in premises]
            best = next((x for x in per if x[1]["karar"] == CEL and x[0].get("id") == "OCRE"), None) or \
                   next((x for x in per if x[1]["karar"] == DEST), None) or \
                   next((x for x in per if x[1]["karar"] == CEL), None) or per[0]
            sonuc.append({"iddia": c, **best[1], "kaynak": best[0].get("kaynak"), "kaynak_id": best[0].get("id"), "kaynak_etiket": best[0].get("etiket")})
        if not sonuc: continue
        ks = [s["karar"] for s in sonuc]
        k = CEL if CEL in ks else YET if YET in ks else DEST
        it["dogrulama"] = {"karar": k, "rozet": ROZET[k], "iddialar": sonuc}
        if k == DEST:
            src = next(s for s in sonuc if s["karar"] == DEST)
            h = next((x for x in hits if x["id"] == src["kaynak_id"]), None)
            if h and oto:
                n = hits.index(h) + 1
                it["kaynaklar"] = [{"no": n, "id": h["id"], "etiket": h["etiket"], "kaynak": h["kaynak"], "sayfa": rag.sayfa(h["kaynak"])}]
                it["etiket"] = h["etiket"]
            it["durum"] = "doğrulandı"
        elif k == YET:
            it["durum"] = "kaynaksız/doğrulanmadı"; it["etiket"] = "kaynaksız — doğrulanmadı"
            if oto: it["kaynaklar"] = []   # 'otomatik atıf' kaldırıldı
        else:
            it["durum"] = "çelişik"; it["etiket"] = "kaynakla çelişiyor"
            bad = next(s for s in sonuc if s["karar"] == CEL)
            if oto: it["kaynaklar"] = []
            yeni_cel.append({"konu": f"doğrulama — {a['ajan']} iddiası kaynakla çelişiyor", "kaynak_turu": "doğrulama",
                             "taraflar": [{"ajan": a["ajan"], "metin": bad["iddia"], "etiket": "ajan iddiası", "kaynaklar": []},
                                          {"ajan": "kaynak", "metin": f"{bad['neden']}", "etiket": bad.get("kaynak_etiket") or "",
                                           "kaynaklar": [{"no": None, "id": bad.get("kaynak_id"), "etiket": bad.get("kaynak_etiket") or "", "kaynak": bad.get("kaynak") or ""}]}]})
    a["dogrulama_celiski"] = yeni_cel
    return a

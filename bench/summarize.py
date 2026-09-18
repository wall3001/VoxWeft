"""汇总 results/*.json → 一张表。python summarize.py"""
import json, glob, os, statistics as st
from paths import RESULTS as R, MANIFEST
rows = {}
for f in sorted(glob.glob(f"{R}/*.json")):
    b = os.path.basename(f)
    if b.startswith(("mt_", "tts_")): continue
    d = json.load(open(f)); m = d["model"]
    for lg in ("vi", "km", "lo"):
        it = [x for x in d["items"] if x["lang"] == lg]
        if not it: continue
        rows.setdefault(m, {})[lg] = dict(
            n=len(it), mean=st.mean(x["cer"] for x in it), med=st.median(x["cer"] for x in it),
            wer=(st.mean(x["wer"] for x in it) if it[0]["wer"] is not None else None),
            rtf=st.mean(x["rtf"] for x in it))
hdr = f"{'模型':14s} | {'vi CER 均/中':>15s} {'vi WER':>7s} | {'km CER 均/中':>15s} | {'lo CER 均/中':>15s} | RTF"
print(hdr); print("-" * len(hdr))
for m, r in rows.items():
    def c(lg):
        x = r.get(lg); return f"{x['mean']:6.1%}/{x['med']:6.1%}" if x else "—"
    w = r.get("vi", {}).get("wer"); w = f"{w:6.1%}" if w else "—"
    rtf = st.mean(x["rtf"] for x in r.values())
    print(f"{m:14s} | {c('vi'):>15s} {w:>7s} | {c('km'):>15s} | {c('lo'):>15s} | {rtf:.2f}")

print("\n== MT (chrF, 展台 3 段平行文本)")
for f in sorted(glob.glob(f"{R}/mt_*.json")):
    d = json.load(open(f))
    a = d.get("corpus_zh2lo", {}); b = d.get("corpus_lo2zh", {})
    sec = st.mean(x["sec"] for x in d["ref"]) if d["ref"] else 0
    print(f"   {d['model']:12s}  zh→lo chrF {a.get('chrF','—'):>5}  lo→zh chrF {b.get('chrF','—'):>5}   平均 {sec:.1f}s/段")

for f in sorted(glob.glob(f"{R}/tts_*.json")):
    d = json.load(open(f))
    if not isinstance(d, list) or not d: continue
    print(f"\n== TTS {d[0]['engine']}: 首包中位 {st.median(x['first_s'] for x in d):.2f}s  RTF 中位 {st.median(x['rtf'] for x in d):.2f}")
    if "cer" in d[0]:
        by = {}
        for x in d: by.setdefault(x["lang"], []).append(x["cer"])
        print("   回环 CER: " + "  ".join(f"{k} {st.mean(v):.1%}" for k, v in by.items()))

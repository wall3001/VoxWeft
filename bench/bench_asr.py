"""ASR 开源替代评测 —— 在 voxlab (M2 16GB) 上跑。

用法:
    python bench_asr.py --model sea-private        # 私有基线（不随仓库分发）
    python bench_asr.py --model qwen17             # Qwen3-ASR-1.7B 开源
    python bench_asr.py --model qwen06             # Qwen3-ASR-0.6B 开源
    python bench_asr.py --model whisper-turbo
    python bench_asr.py --model whisper-v3
    可加 --langs vi,km,lo   --limit N   --chunk 30

评分:
  CER  —— 归一化后（NFC、去零宽/空白/标点、小写）的字符级编辑距离 / 参考长度。
          高棉/老挝语参考文本的分词空格不一致，去空格后 CER 才可比。
  WER  —— 只对越南语（拉丁字母、有词边界）额外算。
切块:
  所有模型统一按 ≤chunk 秒切，边界吸附到窗口尾部 4s 内能量最低点。
  这是「各模型最佳情况」的对比，不是生产切分器（那个 6s 强切）的数字。
"""
import argparse, json, os, sys, time, unicodedata, re, glob
import numpy as np
import soundfile as sf
from rapidfuzz.distance import Levenshtein

HOME = os.path.expanduser("~")
BENCH = os.environ.get("VOXWEFT_BENCH_DIR", f"{HOME}/asr_bench")
MODELS = os.environ.get("VOXWEFT_MODEL_DIR", f"{HOME}/.voxweft/models")
SR = 16000

def _local(*cands):
    """voxlab 上 HF 大文件被卡，模型走 ModelScope / GitHub 落到本地目录。"""
    for c in cands:
        hits = glob.glob(os.path.expanduser(c))
        if hits:
            return hits[0]
    return cands[-1]

MODELS = {
    # private SEA fine-tune used as our internal baseline — not distributed.
    # Point this at your own Qwen3-ASR fine-tune if you have one.
    "sea-private":   ("qwen", os.environ.get("VOXWEFT_PRIVATE_ASR", f"{MODELS}/Qwen3-ASR-private")),
    "qwen17":        ("qwen", _local(f"{BENCH}/models/models/Qwen--Qwen3-ASR-1.7B/snapshots/*", f"{BENCH}/models/models/Qwen/Qwen3-ASR-1.7B")),
    "qwen06":        ("qwen", _local(f"{BENCH}/models/models/Qwen--Qwen3-ASR-0.6B/snapshots/*", f"{BENCH}/models/models/Qwen/Qwen3-ASR-0.6B")),
    "whisper-turbo": ("whisper", _local(f"{BENCH}/models/models/mlx-community--whisper-large-v3-turbo/snapshots/*",
                                        f"{BENCH}/models/models/mlx-community/whisper-large-v3-turbo")),
    "whisper-v3":    ("whisper", _local(f"{BENCH}/models/models/mlx-community--whisper-large-v3-mlx-8bit/snapshots/*",
                                        f"{BENCH}/models/models/mlx-community/whisper-large-v3-mlx-8bit")),
    # Meta Omnilingual ASR（Apache-2.0，1600+ 语含 lo/km/my）。CTC 版语言无关，
    # 走 sherpa-onnx（onnxruntime CPU）。这是 SEA-finetuned Qwen3-ASR 最直接的开源替代候选。
    "omni-1b":       ("omni", f"{BENCH}/models/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-int8-2025-11-12"),
    "omni-300m":     ("omni", f"{BENCH}/models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12"),
    # seanghay 的高棉语续训（Apache-2.0，~700h DDD-Cambodia 数据，config 已含 Khmer 标签）
    "qwen06-km":     ("qwen", _local("/tmp/khmer_asr", f"{BENCH}/models/Qwen3-ASR-0.6B-Khmer")),
}
# Qwen3-ASR 的语言标签。SEA fine-tune 用的是小写训练标签；
# 开源版用英文语言名。两套都试：先按模型类型给默认，跑不通再换。
QWEN_LANG_SEA  = {"vi": "vietnamese", "km": "khmer", "lo": "lao", "th": "thai", "zh": "chinese"}
QWEN_LANG_OPEN = {"vi": "Vietnamese", "km": "Khmer", "lo": "Lao", "th": "Thai", "zh": "Chinese"}
WHISPER_LANG   = {"vi": "vi", "km": "km", "lo": "lo", "th": "th", "zh": "zh"}

_ZW = dict.fromkeys(map(ord, "​‌‍﻿⁠"), None)

def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).translate(_ZW)
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("P")
                and not unicodedata.category(ch).startswith("S"))
    return re.sub(r"\s+", "", s).lower()

def norm_words(s: str) -> list[str]:
    s = unicodedata.normalize("NFC", s).translate(_ZW)
    s = "".join(ch if not unicodedata.category(ch).startswith("P") else " " for ch in s)
    return s.lower().split()

def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    return Levenshtein.distance(r, h) / max(1, len(r))

def wer(ref: str, hyp: str) -> float:
    r, h = norm_words(ref), norm_words(hyp)
    return Levenshtein.distance(r, h) / max(1, len(r))

def chunks(audio: np.ndarray, max_s: float, snap_s: float = 4.0):
    """≤max_s 的块，边界吸附到尾部 snap_s 窗口内 RMS 最低的 32ms 帧。"""
    n = len(audio); out = []; i = 0
    fr = 512
    while i < n:
        j = min(n, i + int(max_s * SR))
        if j < n:
            lo = max(i + fr, j - int(snap_s * SR))
            best, best_rms = j, 1e9
            for k in range(lo, j, fr):
                r = float(np.sqrt(np.mean(audio[k:k+fr] ** 2)))
                if r < best_rms: best_rms, best = r, k
            j = best
        out.append(audio[i:j]); i = j
    return out

def prod_chunks(audio: np.ndarray, sil_thresh: float = 0.035, adaptive: bool = False, min_sil: float = 0.45, max_utt: float = 6.0):
    """用生产切分器切：逐帧 RMS 静音判据 + 6s 强切（ws_server 默认 VOXWEFT_MAX_UTTERANCE_S=6.0）。"""
    import asyncio
    from voxweft.runtime.segmenter import IncrementalSegmenter, FRAME_SAMPLES
    segs = []
    seg = IncrementalSegmenter(lambda s: segs.append(s.audio), max_utterance_s=max_utt, use_smart_turn=False,
                               sil_thresh=sil_thresh, adaptive=adaptive, min_silence_s=min_sil)
    for i in range(0, len(audio) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        seg.feed(audio[i:i + FRAME_SAMPLES])
    asyncio.new_event_loop().run_until_complete(seg.flush())
    return segs

def load_asr(kind: str, path: str):
    if kind == "qwen":
        from mlx_audio.stt.utils import load_model
        m = load_model(path)
        supported = {s.lower() for s in (getattr(m.config, "support_languages", None) or [])}
        def run(seg, lang, is_sea):
            table = QWEN_LANG_SEA if is_sea else QWEN_LANG_OPEN
            name = table[lang]
            # 开源版不认 lo/km/my：硬塞一个没见过的标签是 OOD，退回自动检测
            # （模型自己吐 "language X<asr_text>..."，我们只取正文）。
            hint = name if (is_sea or name.lower() in supported) else None
            r = m.generate(seg, language=hint)
            t = getattr(r, "text", r)
            t = str(t)
            _, sep, tail = t.partition("<asr_text>")
            return (tail if sep else t).strip()
        return run
    elif kind == "omni":
        import sherpa_onnx
        d = path
        rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
            model=f"{d}/model.int8.onnx", tokens=f"{d}/tokens.txt", num_threads=6)
        def run(seg, lang, is_sea):
            st = rec.create_stream()
            st.accept_waveform(sample_rate=SR, waveform=np.ascontiguousarray(seg))
            rec.decode_stream(st)
            return st.result.text.strip()
        return run
    else:
        import mlx_whisper
        def run(seg, lang, is_sea):
            r = mlx_whisper.transcribe(seg, path_or_hf_repo=path, language=WHISPER_LANG[lang],
                                       task="transcribe", fp16=True, condition_on_previous_text=False,
                                       temperature=0.0, no_speech_threshold=0.6)
            return r["text"].strip()
        return run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--langs", default="vi,km,lo")
    ap.add_argument("--limit", type=int, default=0, help="每语种最多跑 N 条")
    ap.add_argument("--chunk", type=float, default=30.0)
    ap.add_argument("--seg", default="fixed", choices=["fixed", "prod"],
                    help="fixed=30s 能量吸附切块（各模型最佳情况）；prod=生产 IncrementalSegmenter（6s 强切，开源版实际路径）")
    ap.add_argument("--tag", default="", help="结果文件后缀")
    ap.add_argument("--sil-thresh", type=float, default=0.035, help="prod 切分器的 RMS 静音门限（生产默认 0.035）")
    ap.add_argument("--adaptive", action="store_true", help="prod 切分器用自适应门限（噪声底×5，夹在 0.006–sil_thresh）")
    ap.add_argument("--min-sil", type=float, default=0.45); ap.add_argument("--max-utt", type=float, default=6.0)
    a = ap.parse_args()
    kind, path = MODELS[a.model]
    is_sea = a.model == "sea-private"
    langs = a.langs.split(",")

    mani = json.load(open(f"{BENCH}/manifest.json"))
    items = [x for x in mani if x["lang"] in langs]
    if a.limit:
        keep = []
        for lg in langs:
            keep += [x for x in items if x["lang"] == lg][: a.limit]
        items = keep

    print(f"== {a.model} ({kind}) {path}", flush=True)
    t0 = time.perf_counter(); run = load_asr(kind, path)
    print(f"   load {time.perf_counter()-t0:.1f}s", flush=True)
    # 预热
    run(np.zeros(SR * 2, dtype=np.float32) + 1e-4, langs[0], is_sea)

    results = []; per_lang = {}
    for it in items:
        audio, sr = sf.read(f"{BENCH}/wav/{it['wav']}", dtype="float32")
        assert sr == SR
        if audio.ndim > 1: audio = audio.mean(axis=1)
        segs = prod_chunks(audio, a.sil_thresh, a.adaptive, a.min_sil, a.max_utt) if a.seg == "prod" else chunks(audio, a.chunk)
        hyps = []; wall = 0.0
        for s in segs:
            t1 = time.perf_counter()
            try:
                h = run(s, it["lang"], is_sea)
            except Exception as e:
                h = ""; print(f"   ! {it['id']} chunk fail: {type(e).__name__}: {e}", flush=True)
            wall += time.perf_counter() - t1
            hyps.append(h)
        hyp = " ".join(hyps)
        c = cer(it["ref"], hyp)
        w = wer(it["ref"], hyp) if it["lang"] == "vi" else None
        rtf = wall / (len(audio) / SR)
        rec = {"id": it["id"], "lang": it["lang"], "dur_s": round(len(audio)/SR, 1), "n_chunks": len(segs),
               "avg_seg_s": round(float(np.mean([len(x)/SR for x in segs])), 2) if segs else 0,
               "cer": round(c, 4), "wer": (round(w, 4) if w is not None else None), "rtf": round(rtf, 3),
               "wall_s": round(wall, 1), "hyp": hyp}
        results.append(rec)
        pl = per_lang.setdefault(it["lang"], {"cer": [], "wer": [], "rtf": [], "dur": 0.0})
        pl["cer"].append(c); pl["rtf"].append(rtf); pl["dur"] += len(audio)/SR
        if w is not None: pl["wer"].append(w)
        print(f"   {it['id']:7s} {rec['dur_s']:6.1f}s  CER {c:6.1%}" + (f"  WER {w:6.1%}" if w is not None else "          ")
              + f"  RTF {rtf:.2f}  | {hyp[:60]}", flush=True)

    print(f"\n== 汇总 {a.model}")
    summ = {}
    for lg, pl in per_lang.items():
        s = {"n": len(pl["cer"]), "cer_mean": round(float(np.mean(pl["cer"])), 4),
             "wer_mean": (round(float(np.mean(pl["wer"])), 4) if pl["wer"] else None),
             "rtf_mean": round(float(np.mean(pl["rtf"])), 3), "dur_s": round(pl["dur"])}
        summ[lg] = s
        print(f"   {lg}: n={s['n']}  CER {s['cer_mean']:.1%}" + (f"  WER {s['wer_mean']:.1%}" if s["wer_mean"] is not None else "")
              + f"  RTF {s['rtf_mean']:.2f}  ({s['dur_s']}s 音频)")
    os.makedirs(f"{BENCH}/results", exist_ok=True)
    json.dump({"model": a.model + a.tag, "path": path, "chunk_s": a.chunk, "seg": a.seg, "summary": summ, "items": results},
              open(f"{BENCH}/results/{a.model}{a.tag}.json", "w"), ensure_ascii=False, indent=1)
    print(f"   -> results/{a.model}{a.tag}.json")

if __name__ == "__main__":
    main()

"""翻译候选评测 —— voxlab (M2 16GB)，mlx-lm。

    python bench_mt.py --model qwen3-4b | qwen3-8b | hy-mt2 | milmmt | gemma4-e4b

数据:
  zh↔lo  展台样例 3 段平行文本（有参考）→ chrF2 / chrF++（sacrebleu）
  vi→zh / km→zh / zh→vi  取 ASR 参考文本前几句做源文（无参考）→ 存输出人工评
统一 greedy 解码（temp=0，repetition_penalty=1.05），保证可复现；
各模型 prompt 按官方模板：Hy-MT2 chat 模板，MiLMMT 裸补全，其余 chat 指令。
"""
import argparse, json, os, re, time, glob
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
from sacrebleu.metrics import CHRF

HOME = os.path.expanduser("~"); BENCH = os.environ.get("VOXWEFT_BENCH_DIR", f"{HOME}/asr_bench"); MODELS = os.environ.get("VOXWEFT_MODEL_DIR", f"{HOME}/.voxweft/models")

def _local(*c):
    for x in c:
        h = glob.glob(os.path.expanduser(x))
        if h: return h[0]
    return c[-1]

M = f"{BENCH}/models/models"
MODELS = {
    "qwen3-4b":   ("chat",   f"{MODELS}/Qwen3-4B-Instruct-2507-4bit"),
    "qwen3-8b":   ("chat",   _local(f"{BENCH}/relay/mlx-community--Qwen3-8B-4bit", f"{M}/mlx-community--Qwen3-8B-4bit/snapshots/*")),
    "hy-mt2":     ("hymt",   _local(f"{M}/mlx-community--Hy-MT2-1.8B-8bit/snapshots/*")),
    "milmmt":     ("milmmt", _local(f"{BENCH}/relay/mlx-community--MiLMMT-46-4B-v0.1-6bit", f"{M}/mlx-community--MiLMMT-46-4B-v0.1-6bit/snapshots/*")),
    "gemma4-e4b": ("chat",   _local(f"{BENCH}/relay/mlx-community--gemma-4-E4B-it-4bit", f"{M}/mlx-community--gemma-4-E4B-it-4bit/snapshots/*")),
}
# 各模板用的语言名
EN = {"zh": "Chinese", "lo": "Lao", "vi": "Vietnamese", "km": "Khmer", "th": "Thai"}
ZH = {"zh": "中文", "lo": "老挝语", "vi": "越南语", "km": "高棉语", "th": "泰语"}
MIL = {"zh": "Chinese (Simplified)", "lo": "Lao", "vi": "Vietnamese", "km": "Khmer", "th": "Thai"}

def build_prompt(kind, tok, src, tgt, text):
    if kind == "milmmt":
        return f"Translate this from {MIL[src]} to {MIL[tgt]}:\n{MIL[src]}: {text}\n{MIL[tgt]}:"
    if kind == "hymt":
        user = (f"Translate the following text into {EN[tgt]}. Note that you should only output "
                f"the translated result without any additional explanation:\n\n{text}")
    else:
        user = f"将以下{ZH[src]}文本翻译成{ZH[tgt]}。只输出译文，不要任何解释或注释。\n\n{text}"
    msgs = [{"role": "user", "content": user}]
    kw = {}
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

_MIL_STOP = re.compile(r"^(Translate this from|(Chinese \(Simplified\)|Lao|Vietnamese|Khmer|Thai|English|Burmese)\s*:)")

def clean(kind, out):
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    if kind == "milmmt":
        # 裸补全：源文有换行时模型也按行出译文，所以不能只取第一行；
        # 只在它开始编造下一个 prompt（"Translate this from…" / "Lao:"）时截断。
        keep = []
        for ln in out.split("\n"):
            if _MIL_STOP.match(ln.strip()):
                break
            keep.append(ln)
        out = "\n".join(keep)
    return out.strip()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--max-tokens", type=int, default=400)
    a = ap.parse_args(); kind, path = MODELS[a.model]

    mani = json.load(open(f"{BENCH}/manifest.json"))
    lo = [x for x in mani if x["lang"] == "lo"]
    def head(s, n=2):
        parts = re.split(r"(?<=[.!?。！？])\s+", s.strip()); return " ".join(parts[:n])
    noref = [("vi", "zh", head(x["ref"])) for x in mani if x["lang"] == "vi"][:3] + \
            [("km", "zh", head(x["ref"])) for x in mani if x["id"] in ("km_07", "km_10", "km_04")] + \
            [("zh", "vi", x["ref_zh"]) for x in lo[:2]]

    print(f"== {a.model} {path}", flush=True)
    t0 = time.perf_counter(); model, tok = load(path); print(f"   load {time.perf_counter()-t0:.1f}s", flush=True)
    sampler = make_sampler(temp=0.0); rp = [make_repetition_penalty(1.05)]
    # 预热
    generate(model, tok, prompt=build_prompt(kind, tok, "zh", "vi", "你好。"), max_tokens=8, sampler=sampler, verbose=False)

    def tr(src, tgt, text):
        p = build_prompt(kind, tok, src, tgt, text)
        t1 = time.perf_counter()
        out = generate(model, tok, prompt=p, max_tokens=a.max_tokens, sampler=sampler, logits_processors=rp, verbose=False)
        return clean(kind, out), time.perf_counter() - t1

    chrf = CHRF(word_order=0); chrfpp = CHRF(word_order=2)
    res = {"model": a.model, "path": path, "ref": [], "noref": []}
    for d_src, d_tgt, key_src, key_ref in (("zh", "lo", "ref_zh", "ref"), ("lo", "zh", "ref", "ref_zh")):
        hyps, refs = [], []
        for x in lo:
            hyp, dt = tr(d_src, d_tgt, x[key_src]); hyps.append(hyp); refs.append(x[key_ref])
            res["ref"].append({"dir": f"{d_src}->{d_tgt}", "id": x["id"], "hyp": hyp, "ref": x[key_ref], "sec": round(dt, 2),
                               "chrf": round(chrf.sentence_score(hyp, [x[key_ref]]).score, 1)})
            print(f"   {d_src}->{d_tgt} {x['id']}  chrF {res['ref'][-1]['chrf']:5.1f}  {dt:5.1f}s | {hyp[:70]}", flush=True)
        c = chrf.corpus_score(hyps, [refs]).score; cpp = chrfpp.corpus_score(hyps, [refs]).score
        res[f"corpus_{d_src}2{d_tgt}"] = {"chrF": round(c, 1), "chrF++": round(cpp, 1)}
        print(f"   → {d_src}->{d_tgt} corpus chrF {c:.1f}  chrF++ {cpp:.1f}", flush=True)
    for src, tgt, text in noref:
        hyp, dt = tr(src, tgt, text)
        res["noref"].append({"dir": f"{src}->{tgt}", "src": text, "hyp": hyp, "sec": round(dt, 2)})
        print(f"   {src}->{tgt} {dt:5.1f}s | {text[:40]} ⇒ {hyp[:70]}", flush=True)
    os.makedirs(f"{BENCH}/results", exist_ok=True)
    json.dump(res, open(f"{BENCH}/results/mt_{a.model}.json", "w"), ensure_ascii=False, indent=1)
    print(f"   -> results/mt_{a.model}.json")

if __name__ == "__main__":
    main()

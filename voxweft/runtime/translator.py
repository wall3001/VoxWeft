"""M3 hybrid translator — Qwen3-32B (mlx-lm) + NLLB, glossary-aware.

Routing strategy (direction-gated):
  lo→zh (and any SEA→zh): Qwen3-32B + glossary prompt  (verified superior)
  zh→lo: Qwen3-32B; fall back to NLLB if QA fails (llama 4B looped on zh→lo)
  SEA↔SEA (non-zh): route through zh bridge (Qwen3 two-hop) OR NLLB direct

Instantiated by the voxweft runtime server in place of a single NLLBTranslator.
"""
import time
import json
import logging
import re
from pathlib import Path

import numpy as np

log = logging.getLogger("voxweft.translator")

LANG_ALIASES = {
    "zh": "zho_Hans", "zh-Hans": "zho_Hans", "zh-Hant": "zho_Hant",
    "lo": "lao_Laoo", "lao": "lao_Laoo", "km": "khm_Khmr", "khmer": "khm_Khmr",
    "my": "mya_Mymr", "burmese": "mya_Mymr", "th": "tha_Thai", "thai": "tha_Thai",
    "vi": "vie_Latn", "viet": "vie_Latn", "id": "ind_Latn", "ms": "zsm_Latn",
    "en": "eng_Latn",
    # flores codes pass through
    "zho_Hans": "zho_Hans", "lao_Laoo": "lao_Laoo", "khm_Khmr": "khm_Khmr",
    "mya_Mymr": "mya_Mymr", "tha_Thai": "tha_Thai", "vie_Latn": "vie_Latn",
}

# For prompting the LLM. Kept in one place: router.LANG_DISPLAY.
from voxweft.runtime.router import LANG_DISPLAY as LANG_TAG, normalize as _normalize


def resolve_lang(code: str) -> str:
    return _normalize(code)


# ── 中文数字串 → 阿拉伯数字 ──────────────────────────────────────────────
#
# 实测：中→老开场白第一句「二零二六年」被 NLLB 译成「ປີ 226」——年份少了
# 一位。它对「十一个国家」这种带位值的写法没问题（正确出 11），坏的只是
# 把零当占位符的连写（年份 / 房号 / 电话号）。ASR 吐的正是这种连写。
#
# 所以只动连写串，带十/百/千的算术写法原样放过（「二十一世纪」不能变 21）。
_CJK_DIGIT = {"〇": "0", "零": "0", "一": "1", "二": "2", "三": "3", "四": "4",
              "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
_CJK_DIGIT_RUN = re.compile(r"[〇零一二三四五六七八九]{2,}")


def normalize_cjk_digits(text: str) -> str:
    def _sub(m):
        run = m.group(0)
        # 没有零的短串更可能是列举（「三四个」「五六十」）而不是数字串。
        # 只在出现 零/〇（纯占位写法的标志）或长度 ≥4（年份）时才转。
        if "零" not in run and "〇" not in run and len(run) < 4:
            return run
        return "".join(_CJK_DIGIT[c] for c in run)
    return _CJK_DIGIT_RUN.sub(_sub, text)


# 「完美和完美」这一类：同一个词被连词/顿号隔开复读。
#
# 只认带连词的形状。正常叠词（研究研究、一个一个、看看）是紧挨着的、不带
# 连词，不能误杀；而「X和X」在中文里没有正常用法，判死即可。尾部的负向
# 预查是防「老挝和老挝人民民主共和国」这种后词更长的真句子被切。
_REPEAT_WITH_CONNECTOR = re.compile(
    r"(?P<w>[一-鿿]{2,6})\s*(?:和|与|及|跟|同|、|，|,)\s*(?P=w)"
    r"(?![一-鿿])")

# 同一个形状的老挝语版本：NLLB 把「女士们，先生们」译成
# 「ສະຫາຍ ແລະ ສະຫາຍ」（同志和同志）。连词是 ແລະ，词间有空格，
# 所以上面那条中文规则认不出来。老挝文没有大小写也没有屈折，重复词
# 判定比中文还干净。
_LAO_REPEAT_WITH_CONNECTOR = re.compile(
    r"(?P<w>[ກ-ໝ][ກ-໿]{1,15})\s*(?:ແລະ|,|，|、)\s*(?P=w)"
    r"(?![ກ-໿])")


# 东南亚四种文字。它们彼此之间不存在合法混排——不像拉丁字母那样会作为
# 专名嵌进别的文字里。所以「泰文译文里出现老挝字母」可以判得很死，
# 见 _is_clean_translation。
_SEA_SCRIPTS = ("tha", "lao", "khm", "mya")


def _cjk_ratio(s: str) -> float:
    """汉字占比。用来决定复读检测走哪一套，见 _is_clean_translation。"""
    body = [c for c in s if c.isalpha()]
    if not body:
        return 0.0
    return sum(1 for c in body if "一" <= c <= "鿿") / len(body)


def _has_consecutive_repeat(s: str, min_unit: int = 3, times: int = 3) -> bool:
    """有没有「同一片段首尾相接连续出现 times 次以上」。

    这是解码循环的真实形状（"ที่ส่งผลต่อการพัฒนา…" 一路接到 max_tokens），
    区别于同一个词在长句里正常地出现两次。

    两个参数都调过：
      min_unit=3  再短（2）会打到 "haha"、越南语声调叠写这类正常构词。
      times=3     叠词在东南亚语言里是**语法**（negara-negara「诸国」、
                  ๆ 重复符），说两遍完全正常；说三遍才是机器坏了。
                  这一条是把印尼语/马来语复数放出来的关键。
    """
    t = re.sub(r"\s+", "", s)
    n = len(t)
    if n < min_unit * times:
        return False
    for w in range(min_unit, min(41, n // times + 1)):
        i = 0
        while i + w * times <= n:
            unit = t[i:i + w]
            k = 1
            while (i + (k + 1) * w <= n
                   and t[i + k * w:i + (k + 1) * w] == unit):
                k += 1
            if k >= times:
                return True
            i += max(1, k) * w
    return False


class HybridTranslator:
    def __init__(self, llm_model_dir: str, glossary_path: str,
                 use_llm_zh_to_lo: bool = True, max_tokens: int = 256):
        self._llm_dir = llm_model_dir
        # 现场逃生开关（launchd: VOXWEFT_USE_LLM_ZH_TO_SEA=1）。
        #
        # 打开后连老挝/高棉/缅甸也交给 LLM，即无视 _NLLB_LANGS 查表。
        # 留着它是因为展台上万一 NLLB 出问题（模型没加载、显存被挤），
        # 改一个环境变量重启服务就能换路，不用改代码——但默认是关的：
        # 这几门语言 4B 译不稳，见 _NLLB_LANGS 的注释。
        self._force_llm_for_nllb_langs = use_llm_zh_to_lo
        self._max_tokens = max_tokens
        self._nllb = None
        self._lm = None
        self._tok = None
        self._glossary = self._load_glossary(glossary_path)
        self._loaded_llm = False

    # 目标语言前缀 → 该语向的专名钉表键。见 _pin_terms。
    #
    # 老挝语钉老挝文，高棉/缅甸钉**拉丁文**——不是偷懒，是实测结论，
    # 理由写在 _pin_terms 的 docstring 里。
    _PIN_TABLES = {"lao_": "zh2lo_pin",
                   "khm_": "zh2khm_pin",
                   "mya_": "zh2mya_pin"}

    @staticmethod
    def _load_glossary(path: str) -> dict:
        """→ {"lo2zh": …, "zh2lo": …, "zh2lo_pin": …, "zh2khm_pin": …,
              "zh2mya_pin": …, "zh2tha": …}

        旧版 glossary.json 是平铺的 SEA→zh 单表。没有方向键的文件一律按
        lo2zh 读，所以换不换文件都能跑。

        新增的键一律 `or {}` 兜底：这个文件是给人改的，缺一个键不该让
        整条翻译链路退化——少一张表只是少一层约束，不是故障。
        """
        keys = ("lo2zh", "zh2lo", "zh2lo_pin", "zh2khm_pin", "zh2mya_pin",
                "zh2tha")
        empty = {k: {} for k in keys}
        try:
            raw = json.loads(Path(path).read_text())
        except Exception:
            return empty
        if not isinstance(raw, dict):
            return empty
        if "lo2zh" in raw or "zh2lo" in raw:
            return {k: dict(raw.get(k) or {}) for k in keys}
        return {**empty,
                "lo2zh": {k: v for k, v in raw.items()
                          if not k.startswith("_")}}

    def _get_nllb(self):
        if self._nllb is None:
            import os as _os
            from voxweft.runtime.translator_nllb import NLLBTranslator
            dev = _os.environ.get("VOXWEFT_NLLB_DEVICE", "mps")
            self._nllb = NLLBTranslator(
                _os.environ.get("VOXWEFT_NLLB_DIR", _os.path.expanduser("~/.voxweft/models/nllb-200-1.3B")),
                device=dev, num_beams=4, no_repeat_ngram=4)
            self._nllb.load()
        return self._nllb

    def _get_llm(self):
        if not self._loaded_llm:
            from mlx_lm import load as mlx_load
            from mlx_lm import generate as mlx_generate
            self._generate = mlx_generate
            self._lm, self._tok = mlx_load(self._llm_dir)
            self._loaded_llm = True
        return self._lm, self._tok

    def warmup(self):
        """Force-load the LLM so the first real turn doesn't pay cold-start
        (load + first-generation) latency. Called once at session setup."""
        if not self._loaded_llm:
            self._get_llm()

    def _generate(self, *a, **k):
        raise RuntimeError("init via _get_llm")

    def _llm_glossary_table(self, src: str, tgt: str) -> dict:
        """这个语向该用哪张术语表，没有就返回空。

        表名带语言不是装饰——zh2lo 里全是老挝文，注进 zh→泰 的 prompt
        就等于命令模型在泰语句子里写老挝字（下面 _translate_llm 里有
        实测的病句）。所以宁可查不到表，也不给一张错的表。
        """
        if src.startswith("zho_"):
            name = {"tha_": "zh2tha", "lao_": "zh2lo"}.get(tgt[:4])
            return self._glossary.get(name, {}) if name else {}
        if src.startswith("lao_") and tgt.startswith("zho_"):
            return self._glossary.get("lo2zh", {})
        return {}

    def _glossary_block(self, pairs: dict) -> str:
        if not pairs:
            return ""
        lines = [f"{k}={v}" for k, v in pairs.items()]
        return "术语对照（必须使用指定译名，不得改写）：\n" + "\n".join(lines)

    def _translate_llm(self, text: str, src: str, tgt: str) -> dict:
        lm, tok = self._get_llm()
        src_name = LANG_TAG.get(src, src)
        tgt_name = LANG_TAG.get(tgt, tgt)
        t0 = time.perf_counter()
        # 表的方向要跟翻译方向一致，否则等于把老挝语词表喂给中文输入，
        # 一条都对不上。
        #
        # 但**方向对还不够，语言也得对**。这两张表叫 lo2zh / zh2lo，名副其实
        # 只管中↔老：里面全是老挝文写法（万象=ວຽງຈັນ）。原来不看语言一律注入，
        # 于是 zh→泰 的 prompt 里也挂着「必须使用指定译名，不得改写：万象=
        # ວຽງຈັນ」——等于命令模型在泰语句子里写老挝字。实测结果正是如此：
        #
        #     期待明年在万象再次相聚
        #     → หวังว่าจะได้พบกันอีกที่วຽງຈັນในปีหน้า
        #                          ~~~~~~ 老挝文，混在泰文句中
        #
        # 这不是模型跑飞，是我们把它按在那儿写的。词表只在中↔老之间注入，
        # 其余语向不给术语表——宁可少一层约束，也不给一条错的约束。
        #
        # 别的语向要术语表的话，得各自建各自的，不能复用这一张。
        # 现在 zh→泰 就有了自己的 zh2tha（泰文写法），选表逻辑收在
        # _llm_glossary_table 里，查不到就不给表。
        gb = self._glossary_block(self._llm_glossary_table(src, tgt))
        sys_p = (f"你是专业{src_name}-{tgt_name}同传译员。只输出译文，不要解释。")
        if gb:
            sys_p += "\n" + gb

        # 用模型自己的 chat template，不要填空式 prompt。
        #
        # 这里原来是给 Qwen3-32B-4bit 写的 completion 式 prompt（"…译文："），
        # 那确实是个 base 模型、没有 chat_template。但现在跑的是
        # **Qwen3-4B-Instruct**，它带 chat_template.jinja，而填空式 prompt
        # 拼不出 <|im_start|>/<|im_end|> 框架——于是模型**没有停下来的信号**，
        # 一路生成到 max_tokens 为止。实测形状是「前半句译得好好的，后面开始
        # 车轱辘话」：
        #
        #     会议汇聚了中国、老挝、泰国、越南等十一个国家的代表。(26 字)
        #     → 451 字，前 90 字是正确译文，其后 ที่ส่งผลต่อการพัฒนา… 循环到截断
        #
        # 这些输出全部被护栏判退、回落 NLLB，所以屏幕上看不出问题——代价是
        # **zh→泰 75% / 印尼 60% / 马来 60% / 英 45% 的句子根本没用上 LLM**，
        # 而路由表和展台话术都写着「这几个语向 Qwen 强过 NLLB」。等于我们
        # 一直在演示 NLLB，却以为在演示 Qwen。
        #
        # 换成 chat template 之后同一句：77 字 / 350ms（原来 451 字 / 1554ms），
        # 自然停在 <|im_end|>，专名也对了（万象 → เวียงจันทร์，原来是 วานาไพ่）。
        #
        # 注意 chat_stream()（对话模式）早就在用 _apply_chat_template 了，
        # 只有翻译这条路没跟上——两处现在统一。
        msgs = [{"role": "system", "content": sys_p},
                {"role": "user", "content": text}]
        prompt = self._apply_chat_template(tok, msgs)
        out = self._generate(lm, tok, prompt=prompt, max_tokens=self._max_tokens,
                             verbose=False)
        # strip the echo of the instruction + any label before returning
        res = out.strip()
        if "译文：" in res:
            res = res.rsplit("译文：", 1)[-1].strip()
        latency = (time.perf_counter() - t0) * 1000
        return {"text": res, "latency_ms": latency}

    def _translate_nllb(self, text: str, src: str, tgt: str) -> dict:
        # NLLB 吃不进 prompt，术语只能在两头夹：进去之前把中文数字串正规化，
        # 出来之后按对照表校正被原样抄回的术语。中↔老全程走这条路
        # （见 translate() 的 _NLLB_LANGS 查表），所以这是该方向唯一的干预点。
        zh_src = src.startswith("zho_")
        fed = normalize_cjk_digits(text) if zh_src else text
        # 钉专名：老挝 / 高棉 / 缅甸，也就是 _NLLB_LANGS 里那三门。
        # 别的语向走 LLM，不经过这里。
        if zh_src:
            fed = self._pin_terms(fed, tgt)
        result = self._get_nllb().translate(fed, src, tgt)
        out = self._stabilize_repetition(result.get("text", ""))
        # 同样只在中→老这一向做。zh2lo 表里的译名是老挝文，往泰语/印尼语
        # 译文里替换等于插一段老挝字进去——和 _translate_llm 里那个词表注入
        # 是同一个 bug 的另一半，见那边的长注释。
        if zh_src and tgt.startswith("lao_"):
            out = self._enforce_terms(fed, out)
        result["text"] = out
        return result

    def _pin_terms(self, text: str, tgt: str) -> str:
        """把专名在**进 NLLB 之前**换成目标语写法，让模型照抄。

        实测 NLLB 会把「万象」音译成 Wan Chai（香港湾仔）、「琅勃拉邦」
        译成 Wembla，而且是稳定复现的——译文里已经没有中文，出来之后
        再查表是查不到的。但反过来，源句里直接嵌目标语专名时 NLLB 会原样
        抄过去，还会放对位置（「ສູນປະຊຸມໃນວຽງຈັນແລະຫຼວງພະບາງ」），
        所以专名在入口钉死最省事。

        **只放专名和固定搭配**。像「会议」「代表」这种普通词不能钉：
        NLLB 只会把它抄在中文原来的位置上，不跟着目标语的语序走，
        钉一个动词进去就是造病句。普通词留给 _enforce_terms 兜底。

        ── 为什么高棉/缅甸钉的是拉丁文，不是本族文字 ───────────────
        因为试过，本族文字钉不住，而且失败的方式是**换一个国家**：

            缅甸  ဗီယင်ကျန်း(万象) → ဗီယက်နမ်（越南）
                  လွမ်းပရာဘန်(琅勃拉邦) → လန်ဒန်（伦敦）
            高棉  ហ្លួងព្រះបាង(琅勃拉邦) → ហុងកុង（香港）

        钉进去的字串不在 NLLB 的词表里，它就当成噪声重新猜一个形近的
        地名，于是「明年在万象见」变成「明年在越南见」。这比不钉更糟。

        拉丁文两种语言、两个位置全都原样抄过来（វៀងចន្ទន៍→Vientiane、
        ဗီယင်ကျန်း→Vientiane），而且拉丁地名谁都能一眼核对对错——这一点
        对一份没人读得懂高棉文的素材尤其要紧。

        老挝语保持本族文字：那一向是验证过的，ວຽງຈັນ 稳定抄得过去。
        """
        table = next((v for k, v in self._PIN_TABLES.items()
                      if tgt.startswith(k)), None)
        pins = self._glossary.get(table or "", {})
        if not pins or not text:
            return text
        # 长词优先：先钉「中国—东盟博览会」，免得「中国」先把它拆了
        for zh, lo in sorted(pins.items(), key=lambda kv: -len(kv[0])):
            if zh in text:
                text = text.replace(zh, lo)
        return text

    def _enforce_terms(self, src: str, out: str) -> str:
        """zh→SEA：把 NLLB 原样抄回译文的中文术语换成对照表里的译名。

        只处理这一种能确定判定的坏法——NLLB 在低资源方向上对专名常直接
        复制源词。**不做位置插入**：译文里没有这个词时无从知道该插在哪，
        硬插只会造出更难查的错句。
        """
        pairs = self._glossary["zh2lo"]
        if not pairs or not out:
            return out
        # 长词优先，免得「会议」先把「国际会议」拆散
        for zh, tgt in sorted(pairs.items(), key=lambda kv: -len(kv[0])):
            if zh in out and zh in src:
                out = out.replace(zh, tgt)
        return out

    @staticmethod
    def _stabilize_repetition(text: str) -> str:
        """Prevent low-resource decoding loops from reaching the UI/TTS.

        Fragmented streaming turns can make NLLB repeat one token many times
        (for example a repeated Lao noun separated by commas). Preserve normal
        prose, but collapse a run of the same token when it occurs three or
        more times consecutively.
        """
        if not text:
            return text
        # 「完美和完美」：同一个词被连词隔开复读。下面那条规则只认逗号/
        # 空格分隔，中间夹个「和」就漏过去了——这是展台上实际看到的形状。
        prev = None
        while prev != text:
            prev = text
            text = _REPEAT_WITH_CONNECTOR.sub(r"\g<w>", text)
            text = _LAO_REPEAT_WITH_CONNECTOR.sub(r"\g<w>", text)
        # First handle the common decoder-loop shape:
        # "token, token, token" or "token token token".
        text = re.sub(
            r"(?P<token>[^\s,，、。.!！?？;；:：]+)"
            r"(?:\s*[,，、]\s*(?P=token)){2,}",
            r"\g<token>",
            text,
        )
        parts = re.split(r"(\s+|[,，、。.!！?？;；:：])", text)
        out: list[str] = []
        last_word = None
        repeat_count = 0
        for part in parts:
            word = part.strip()
            if not word or re.fullmatch(r"[\s,，、。.!！?？;；:：]+", part):
                out.append(part)
                continue
            if word == last_word:
                repeat_count += 1
                if repeat_count >= 2:
                    continue
            else:
                last_word = word
                repeat_count = 1
            out.append(part)
        return "".join(out).strip()

    # ── 对话（agent 模式）────────────────────────────────────────────────────
    #
    # 复用同一个 LLM 实例，不另起 mlx_lm.server。
    # 理由：模型加载要几十秒 + 几 GB 内存；M3 上翻译和对话不会同时高负载
    # （用户要么在说、要么在听），共享一份最划算。而且 warmup() 已经把权重
    # 预加载好了，对话首句能直接受益。
    #
    # 注意 translator 的 _translate_llm 走的是 completion 式 prompt（注释说
    # 32B 是 base 模型没有 chat_template）。4B-Instruct 是有 template 的，
    # 所以这里优先用 apply_chat_template，没有才退回填空式。

    def chat_stream(self, messages: list[dict], max_tokens: int = 200,
                    temperature: float = 0.7):
        """流式生成，逐 token yield（阻塞生成器，调用方丢线程池里跑）。

        mlx_lm 这版的采样参数是 sampler 对象，不是 `temp=`（旧版写法会报
        `generate_step() got an unexpected keyword argument 'temp'`）。
        """
        lm, tok = self._get_llm()
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self._apply_chat_template(tok, messages)
        sampler = make_sampler(temp=temperature)
        for resp in stream_generate(lm, tok, prompt=prompt,
                                    max_tokens=max_tokens, sampler=sampler):
            piece = getattr(resp, "text", None)
            if piece:
                yield piece

    @staticmethod
    def _apply_chat_template(tok, messages: list[dict]) -> str:
        """优先用模型的 chat template；没有就退回手写的填空式 prompt。"""
        if getattr(tok, "chat_template", None):
            try:
                return tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(content)
            elif role == "user":
                parts.append(f"用户：{content}")
            else:
                parts.append(f"助手：{content}")
        parts.append("助手：")
        return "\n".join(parts)

    # 作为**目标语言**时走 NLLB 的语言（FLORES-200 前缀）。
    #
    # 判据是「让 4B 写出这门文字」靠不靠谱，不是「它认不认识」：老挝 /
    # 高棉 / 缅甸在通用 LLM 里语料稀薄，生成端会蒙——出来的东西读着通顺、
    # 意思是错的，而展台上没人当场能发现。NLLB-200 专门在这些语向上训过，
    # 宁可译得平淡也不会自己发挥。
    # 英 / 泰 / 越 / 印尼 / 马来相反，Qwen 直接比 NLLB 强，尤其会议长句——
    # NLLB 会把长句切碎、丢掉从句。
    #
    # **只看目标侧。** 方向是不对称的：zh→lo 要模型生成老挝文（弱项），
    # lo→zh 只要看懂老挝文再生成中文（强项，实测整句 267-310ms）。
    #
    # ⚠️ 和 voxweft.runtime.router.NLLB_LANGS 是同一张表，必须一起改：
    # 那边决定屏幕上通告的路由，这里决定实际调用哪个引擎。
    _NLLB_LANGS = ("lao_", "khm_", "mya_")

    @classmethod
    def _prefers_nllb(cls, code: str) -> bool:
        return code.startswith(cls._NLLB_LANGS)

    # 各语言「正文字符」所在的 Unicode 区段。给 _is_clean_translation 判
    # 「译出来的是不是目标那门文字」用。
    #
    # ⚠️ 泰文 (0E00-0E7F) 和老挝文 (0E80-0EFF) 在码表上紧挨着，字形也像，
    # 正是实测最常撞的一对——老挝语目标吐出泰文，人眼扫过去只觉得「像那么
    # 回事」。分开列就是为了让机器分得出人分不出的这一对。
    _SCRIPT = {
        "zho": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
        "eng": [(0x0041, 0x005A), (0x0061, 0x007A)],
        "ind": [(0x0041, 0x005A), (0x0061, 0x007A)],
        "zsm": [(0x0041, 0x005A), (0x0061, 0x007A)],
        "vie": [(0x0041, 0x005A), (0x0061, 0x007A),
                (0x00C0, 0x024F), (0x1E00, 0x1EFF)],
        "tha": [(0x0E00, 0x0E7F)],
        "lao": [(0x0E80, 0x0EFF)],
        "khm": [(0x1780, 0x17FF)],
        "mya": [(0x1000, 0x109F), (0xAA60, 0xAA7F)],
        # Latin-script Europeans: ASCII + Latin-1/Extended-A for diacritics.
        "deu": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
        "fra": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
        "spa": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
        "ita": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
        "por": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
        "rus": [(0x0400, 0x04FF)],
        "arb": [(0x0600, 0x06FF), (0x0750, 0x077F)],
        # Japanese: kana + CJK ideographs. Korean: Hangul syllables + jamo.
        "jpn": [(0x3040, 0x30FF), (0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
        "kor": [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],
    }

    @classmethod
    def _script_ratio(cls, s: str, code3: str) -> float:
        """正文字符里落在该语言文字系统内的比例。

        标点、数字、空格不进分母——译文里出现 "2026"、逗号、括号都正常，
        算进去只会让短句的比例无意义地抖。
        """
        ranges = cls._SCRIPT.get(code3, [])
        body = [c for c in s if c.isalpha()]
        if not body:
            return 0.0
        return sum(1 for c in body
                   if any(lo <= ord(c) <= hi for lo, hi in ranges)) / len(body)

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> dict:
        src = resolve_lang(src_lang)
        tgt = resolve_lang(tgt_lang)
        src_is_zh = src.startswith("zho_")
        tgt_is_zh = tgt.startswith("zho_")

        # 选引擎：只看**目标语言**。低资源目标（lo/km/my）走专用 MT，其余
        # 任何语对——含 es→ms、de→vi 这类不沾中英文的——都让 LLM 直译，
        # 不经中转（两跳会把误差乘起来）。
        if self._force_llm_for_nllb_langs:
            use_llm = True
        else:
            use_llm = not self._prefers_nllb(tgt)

        # 把选中的引擎打进日志：展台上译文一旦不对劲，第一个要回答的问题
        # 就是「这句是谁译的」。事后翻日志比复现快得多。
        log.info("route %s→%s → %s", src, tgt, "LLM" if use_llm else "NLLB")

        try:
            if use_llm:
                r = self._translate_llm(text, src, tgt)
                res = (r.get("text") or "").strip()
                if not self._is_clean_translation(text, res, src_is_zh,
                                                  tgt_is_zh, tgt):
                    log.info("LLM output degraded (%s→%s len=%d/%d), "
                             "falling back to NLLB",
                             src, tgt, len(res), len(text))
                    return self._translate_nllb(text, src, tgt)
                if res:
                    return r
                # fallthrough for degenerate output
            return self._translate_nllb(text, src, tgt)
        except Exception as e:
            log.warning("LLM translation failed (%s), falling back to NLLB: %s",
                        e, e)
            try:
                return self._translate_nllb(text, src, tgt)
            except Exception as e2:
                return {"text": "", "latency_ms": 0, "error": str(e2)}

    @classmethod
    def _is_clean_translation(cls, src: str, res: str, src_is_zh: bool,
                              tgt_is_zh: bool, tgt: str = "") -> bool:
        """Heuristic guard against LLM drift. Returns False → fall back to NLLB.

        Catches the failure modes seen on M3 with Qwen3-32B AND Qwen3-4B:
          * zh→non-zh: Traditional-char leak, wordy re-statement of the source.
          * non-zh→zh (esp. short/fragmented ASR segments): meta-commentary
            ("由于…", "无法提供"…), prompt echo, SEA-script residue,
            and verbatim phrase repetition (4B loops on short segments).
          * zh→non-zh: **蒙错了文字系统**（见下面 tgt 那一段）。
        """
        if not res or len(res) < 2:
            return False
        # Japanese legitimately contains CJK ideographs; the "Chinese leaked
        # into the target" checks below must not fire on it.
        tgt_uses_cjk = tgt_is_zh or (tgt or "").startswith("jpn_")
        # Source leaked back into a non-CJK target.
        if src_is_zh and not tgt_uses_cjk:
            if any("一" <= ch <= "鿿" for ch in res):
                return False
        if not tgt_uses_cjk:
            # target should be predominately non-CJK; any big CJK run = leak
            cjk = sum(1 for ch in res if "一" <= ch <= "鿿")
            if cjk > max(2, len(res) * 0.05):
                return False

        # 译出来的是不是**目标那门文字**。
        #
        # 上面几条查的都是「中文有没有漏出去」——它们看得见 zh，看不见
        # 「该出印尼文却出了老挝文」。而这正是建底本时真实撞到的两次：
        #
        #   01→马来语「会议汇聚了中国、老挝…」→ 吐出泰文，内容还是别句的
        #   04→印尼语「欢迎各国代表前来实地考察」→ 吐出老挝文，且把「实地
        #              考察」译成了 ສົງຄາມ（战争）
        #
        # 两条都完整通过了旧版全部检查：没有 CJK、不重复、无元评论、长度
        # 正常。因为旧版只拿到 src_is_zh / tgt_is_zh 两个布尔，**结构上就
        # 无法**表达「哪一门」。所以这里要的是 tgt 本身，不是又一条规则。
        #
        # 0.60 而不是 0.95：专有名词、缩写、"2026" 保留原样是正常译法
        # （NLLB 那条路上的 ASEAN / Digital Connectivity Partnership 就是
        # 这样）。低于 6 成基本只能是整段蒙错了语言。
        code3 = (tgt or "")[:3].lower()
        if code3 in cls._SCRIPT and len(res) >= 8:
            r = cls._script_ratio(res, code3)
            if r < 0.60:
                log.info("LLM 蒙错文字系统：%s 只有 %.0f%% 落在 %s 区段",
                         res[:30], r * 100, tgt)
                return False

        # 串进来**另一种** SEA 文字。
        #
        # 上面那条比例检查拦的是整段蒙错，拦不住只错几个字的：实测
        # 05→泰语「明年在万象再会」译成了
        #     หวังว่าจะได้พบกันอีกที่วຽງຈັນในปีหน้า
        # ——句子是泰文，唯独「万象」五个字母写成了老挝文。比例 0.85，
        # 高高在阈值之上，但泰国观众看到的是句子中间插了一段外国字。
        #
        # 这条能判得这么死（>0 即拦），是因为泰 / 老 / 高棉 / 缅甸四种文字
        # 之间不存在合法的混排——不像拉丁字母那样会作为专名嵌进别的文字。
        # 所以不设比例，一个字符都算错。
        if code3 in _SEA_SCRIPTS:
            for other in _SEA_SCRIPTS:
                if other == code3:
                    continue
                if any(any(lo <= ord(ch) <= hi for lo, hi in cls._SCRIPT[other])
                       for ch in res):
                    log.info("LLM 在 %s 译文里混入了 %s 文字：%s",
                             tgt, other, res[:30])
                    return False

        # SEA-script residue in a zh target (lao `ກ-໿`, thai `ก-๙`, khmer `ក-៳`)
        if tgt_is_zh:
            lao = sum(1 for ch in res if "ກ" <= ch <= "໿")
            thai = sum(1 for ch in res if "ก" <= ch <= "๙")
            khmer = sum(1 for ch in res if "ក" <= ch <= "៿")
            if lao + thai + khmer > 2:
                return False
        # Verbatim repetition (4B/32B loop on short/fragmented input, e.g.
        # "從老挝，像" repeated on separate lines). Lines repeat => degenerate.
        lines = [l.strip() for l in res.splitlines() if l.strip()]
        if len(lines) >= 2:
            uniq_lines = set(lines)
            if len(uniq_lines) < len(lines):       # a line repeats
                return False
        # 短串复读：「完美和完美」只有 5 个字，下面那个 5 字滑窗够不着
        # （len < 8 根本不查），而分句流里最常见的退化恰恰是这种短句。
        if _REPEAT_WITH_CONNECTOR.search(res):
            return False
        # 复读检测。**分两套**，因为「5 个字符重复出现」在中文和在拼音文字里
        # 根本不是同一件事。
        #
        # 中文一个字就是一个词，连续 5 个字原样再现基本只能是解码跑飞
        # （"完美和完美"、"…像像像…"），所以对中文保留原来那条很凶的规则：
        # 任意位置重现即判退。
        #
        # 拼音文字里这条必然误伤——5 个字符还不到一个词：
        #
        #   Semoga upacara ini berjalan lancar dan berjaya.   berjalan/berjaya 共享 " berj"
        #   Kính thưa các vị khách quý, các chị em…           越南语冠词 "các " 本来就要出现多次
        #   Selamat datang… perwakilan negara-negara…         印尼语复数就是叠词
        #
        # 三句都是完全正确的译文，全被判成「退化」退回了 NLLB。实测 zh→印尼/
        # 马来/越南 的退化有一多半来自这一条，屏幕上看不出来（NLLB 也能译），
        # 但等于把 LLM 这条路白白关掉了一半。
        #
        # 所以拼音文字这边只抓**连续**重复——真的循环是 "abc abc abc" 这种
        # 首尾相接，而不是同一个词在句子两头各出现一次。
        if len(res) >= 8:
            if _cjk_ratio(res) > 0.3:
                seen = set()
                for i in range(len(res) - 5):
                    unit = res[i:i + 5]
                    if unit in seen:
                        return False
                    seen.add(unit)  # only flag re-appearance, not overlap
            elif _has_consecutive_repeat(res):
                return False
        # Meta-commentary / explanation / prompt-echo markers (any direction).
        META = ("由于", "无法", "如果您", "根据", "用户要求", "需要提供",
                "似乎", "不完整", "意思是", "翻译如下", "以下", "这段话",
                "please", "注:", "备注", "解释", "不能",
                # prompt echo (model regurgitates the instruction, not the
                # translation — seen on fragmented SEA segments with 32B)
                "请把下面的", "只给译文", "不要解释", "同传译员", "术语对照",
                "译文：", "输出译文", "专业")
        if any(m in res for m in META):
            return False
        # Length distortion: a compact translation should not balloon.
        if len(res) > 2.5 * len(src) + 120:
            return False
        return True

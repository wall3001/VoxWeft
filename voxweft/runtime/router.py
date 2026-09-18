"""LanguageRouter — 按「任务 + 语言对」选后端。纯函数，无状态，不碰 I/O。

为什么需要它：展厅演示要覆盖东盟十国，但没有任何单一模型覆盖全部语种。
ASR 有三条路径（自研 SEA-finetuned Qwen3-ASR / Whisper 兜底 / 中文本地），翻译有三条
（Qwen3-LLM / NLLB / 护栏回退），TTS 一条但音色要分角色。这些选择散落在
ws_server 的 if-else 里会失控——集中到这里，加语言只改一张表。

设计约束（都是实测踩出来的，见 m3/ROUTING.md §1）：
  * mlx ASR 只认小写训练标签。传 'lo'/'Lao'/'lao_Laoo' 会静默退化成泰文，
    不报错。所以 normalize 是安全边界，不是顺手做的格式化。
  * zh→lo 方向小 LLM 会蒙错语言（出高棉文/泰文混排），必须 NLLB。
    方向门控在这里表达，不散到调用点。
  * SEA↔SEA 不经中文中转——两跳错误累积且延迟翻倍。
"""
from __future__ import annotations

import logging

log = logging.getLogger("voxweft.router")

# ── 语言码归一 ──────────────────────────────────────────────────────────────
# 三层各有各的坑，这里统一成 FLORES-200 作为内部标准码，出口再各自映射。

# 内部标准码（FLORES-200）
ZH = "zho_Hans"
LAO = "lao_Laoo"
THA = "tha_Thai"
KHM = "khm_Khmr"
MYA = "mya_Mymr"
VIE = "vie_Latn"
ENG = "eng_Latn"
IND = "ind_Latn"
MSA = "zsm_Latn"

# 任意客户端写法 → FLORES-200
LANG_ALIASES = {
    "zh": ZH, "zh-hans": ZH, "zh-hant": ZH, "zho": ZH, "zho_hans": ZH,
    "zho_hant": ZH, "chinese": ZH, "cn": ZH, "mandarin": ZH, "中文": ZH,
    "lo": LAO, "lao": LAO, "lao_laoo": LAO, "laos": LAO, "laotian": LAO,
    "老挝语": LAO, "老挝": LAO,
    "th": THA, "tha": THA, "tha_thai": THA, "thai": THA, "泰语": THA,
    "km": KHM, "khm": KHM, "khm_khmr": KHM, "khmer": KHM, "高棉语": KHM,
    "my": MYA, "mya": MYA, "mya_mymr": MYA, "burmese": MYA,
    "myanmar": MYA, "缅甸语": MYA,
    "vi": VIE, "vie": VIE, "vie_latn": VIE, "vietnamese": VIE, "越南语": VIE,
    "en": ENG, "eng": ENG, "eng_latn": ENG, "english": ENG, "英语": ENG,
    "id": IND, "ind": IND, "ind_latn": IND, "indonesian": IND, "印尼语": IND,
    "ms": MSA, "zsm": MSA, "zsm_latn": MSA, "malay": MSA, "马来语": MSA,
}

# 向用户展示的名字（prompt 里用）
LANG_DISPLAY = {
    ZH: "中文", LAO: "老挝语", THA: "泰语", KHM: "高棉语", MYA: "缅甸语",
    VIE: "越南语", ENG: "英语", IND: "印尼语", MSA: "马来语",
}

# FLORES-200 → mlx SEA-SFT 训练标签（小写）
FLORES_TO_MLX = {
    ZH: "chinese", LAO: "lao", THA: "thai", KHM: "khmer",
    MYA: "burmese", VIE: "vietnamese", ENG: "english",
    IND: "indonesian", MSA: "malay",
}

# SEA-finetuned Qwen3-ASR 微调覆盖的语种（其余走 Whisper）
ASR_80K_LANGS = {LAO, THA, KHM, MYA, VIE, ZH}

# SEA 语系（不含中文）—— 用于判断是否走 NLLB 直译
SEA_LANGS = {LAO, THA, KHM, MYA, VIE, IND, MSA}


def normalize(code: str) -> str:
    """任意写法 → FLORES-200。认不出就原样返回（让下游报错，不静默猜）。"""
    if not code:
        return ZH
    key = str(code).strip()
    hit = LANG_ALIASES.get(key.lower())
    if hit:
        return hit
    # 已经是大写敏感的 FLORES 码
    if key in LANG_DISPLAY:
        return key
    log.warning("unknown language code %r — passing through unchanged", code)
    return key


def is_zh(code: str) -> bool:
    return normalize(code) == ZH


def display(code: str) -> str:
    return LANG_DISPLAY.get(normalize(code), normalize(code))


# ── ASR 路由 ────────────────────────────────────────────────────────────────

class ASRRoute:
    """选中的 ASR 后端 + 该后端要的语言串。"""

    __slots__ = ("backend", "lang", "hotwords")

    def __init__(self, backend: str, lang: str, hotwords: list[str]):
        self.backend = backend   # "qwen3asr" | "whisper"
        self.lang = lang         # 直接喂给后端的串（已归一）
        self.hotwords = hotwords

    def __repr__(self):
        return f"ASRRoute({self.backend}, lang={self.lang!r}, hw={len(self.hotwords)})"


def route_asr(src_lang: str, hotwords: list[str] | None = None,
              default_hotwords: dict | None = None) -> ASRRoute:
    """按源语言选 ASR 后端。

    SEA-finetuned Qwen3-ASR 是我们自研的权重，低资源语系（老/泰/高棉/缅/越）是它的强项——
    这些语言在其他任何开源 ASR 上都是无人区。它没微调过的（印尼/马来/英）
    走 Whisper-Large-v3。
    """
    code = normalize(src_lang)
    hw = list(hotwords or [])
    if not hw and default_hotwords:
        hw = list(default_hotwords.get(code, []))

    if code in ASR_80K_LANGS:
        return ASRRoute("qwen3asr", FLORES_TO_MLX.get(code, "lao"), hw)
    return ASRRoute("whisper", _whisper_lang(code), hw)


def _whisper_lang(code: str) -> str:
    """Whisper 认 ISO-639-1 两字母码，不是 FLORES。"""
    return {
        ENG: "en", IND: "id", MSA: "ms", VIE: "vi",
        THA: "th", LAO: "lo", KHM: "km", MYA: "my", ZH: "zh",
    }.get(code, code)


# ── 翻译路由 ────────────────────────────────────────────────────────────────

class TranslateRoute:
    __slots__ = ("backend", "reason")

    def __init__(self, backend: str, reason: str):
        self.backend = backend   # "llm" | "nllb"
        self.reason = reason

    def __repr__(self):
        return f"TranslateRoute({self.backend}, {self.reason})"


# 不能**生成**得可靠的语言（FLORES-200 前缀），作为目标语言时走 NLLB。
#
# 判据是「让 4B 写出这门文字」靠不靠谱，不是「它认不认识」：老挝 / 高棉 /
# 缅甸在通用 LLM 预训练里语料稀薄，生成端会蒙错语言（实测 gemma3-4b 出
# 高棉文、Qwen3-4B 出泰文混排+循环），NLLB-200 有专门训练。
# 英 / 泰 / 越 / 印尼 / 马来则相反，Qwen 生成明显强过 NLLB，尤其长句从句。
#
# 只对**目标**侧生效。反向（lao→zh）生成的是中文，是 4B 的强项，照走 LLM。
#
# ⚠️ voxweft.runtime.translator.HybridTranslator._NLLB_LANGS 必须和这里保持一致。
# 那边决定**实际**用哪个引擎，这里决定**对外通告**的路由（session_config_ack
# 里的 routes 字符串，会直接显示在展台屏幕的「当前路由」条上）。两份写歪了
# 的话，屏幕上写着 llm、实际跑的是 nllb，观众问起来没法圆。
NLLB_LANGS = ("lao", "khm", "mya")


def _prefers_nllb(code: str) -> bool:
    """code 是 normalize() 之后的 FLORES 码，如 lao_Laoo。"""
    return str(code).lower().startswith(NLLB_LANGS)


def route_translate(src_lang: str, tgt_lang: str,
                    use_llm_translator: bool = True,
                    use_llm_zh_to_sea: bool = False) -> TranslateRoute:
    """按语言对选翻译后端。

    规则（都有实测依据，见 ROUTING.md §1.2）：
      * →老挝/高棉/缅甸：NLLB。小 LLM 生成这些低资源文字会蒙错语言。
      * →中文 / →英泰越印尼马来：Qwen3-LLM（长句、从句明显好过 NLLB），
        护栏失败回退 NLLB。
      * SEA↔SEA：NLLB 直译，不经中文中转。

    `use_llm_zh_to_sea` 是现场逃生开关（VOXWEFT_USE_LLM_ZH_TO_SEA=1）：
    打开后连第一类也交给 LLM。默认关。
    """
    src, tgt = normalize(src_lang), normalize(tgt_lang)
    if src == tgt:
        return TranslateRoute("nllb", "same-lang (NLLB echo)")

    s_zh, t_zh = src == ZH, tgt == ZH

    # SEA ↔ SEA 直译。两跳（SEA→zh→SEA）会错误累积 + 延迟翻倍。
    if not s_zh and not t_zh:
        return TranslateRoute("nllb", "SEA<->SEA direct")

    if not use_llm_translator:
        return TranslateRoute("nllb", "LLM translator disabled")

    # 按**目标语言**查表——方向是不对称的，而且这个不对称是真的：
    #
    #   zh→lo：要模型**生成**老挝文。4B 没有足够的老挝语料，生成时会蒙
    #          （实测 gemma3-4b 出高棉文、Qwen3-4B 出泰文混排+循环）。→ NLLB
    #   lo→zh：只要**看懂**老挝文、**生成中文**。生成端是 4B 的强项，
    #          实测完整句 267-310ms 且质量好。→ LLM
    #
    # 原来这条是 `if s_zh and not t_zh and not use_llm_zh_to_sea`：方向判对了，
    # 但把「哪门语言」漏了——英/泰/越/印尼/马来这些 Qwen 明显更强的目标语言，
    # 被同一个开关一起打到 NLLB 上。一个开关同时管两类相反的语言，怎么设
    # 都是错的，所以方向照旧、语言换成查表。
    if _prefers_nllb(tgt) and not use_llm_zh_to_sea:
        return TranslateRoute("nllb", f"->{tgt} (small LLM unreliable)")

    return TranslateRoute("llm", "zh involved, LLM has the prior")


# ── TTS 路由 ────────────────────────────────────────────────────────────────
#
# ⚠️ 不要给合成文本加音色描述前缀。
#
# 曾经这样做过：给 TTS 文本拼上 `(一位沉稳清晰的播音员)`，本意是当隐式音色
# 条件。当时的注释写「caption 字未被读出」——**那是错的**。实测（ASR 回环）：
#
#   不带前缀  2.56s  →  「欢迎参加东盟数字经济会议。」
#   带前缀    4.80s  →  「阿千一位沉稳清晰的播音员，欢迎参加东盟数字经济会议。」
#
# 即 VoxCPM2 会把括号里的描述当正文念出来。展厅里听众会先听到一句莫名其妙的
# 「一位沉稳清晰的播音员」，非常尴尬。
#
# 现在统一不加前缀。若将来需要区分音色，正确做法是给 VoxCPM2 换 reference
# 音频（音色克隆），而不是在文本里加描述。

# 角色名保留 —— 接口上仍然传 role，但当前不做音色区分。
# 保留它是为了未来接音色克隆时有个现成的挂载点。
VOICE_ROLES = ("interpreter", "agent_default", "agent_warm")
DEFAULT_VOICE = "interpreter"


def voice_caption(role: str | None) -> str:
    """历史接口。**永远返回空串** —— 见上面关于前缀会被念出的说明。

    保留这个函数而不是删掉，是为了让调用点不必改动；如果将来要做音色克隆，
    在这里换成「解析出 reference latents」即可。
    """
    return ""


def apply_voice(text: str, role: str | None = None) -> str:
    """给合成文本套上音色条件。

    当前是**恒等变换**（不加任何前缀）——见文件上方说明。保留函数是为了
    未来接音色克隆：那时改这里，调用点不用动。
    """
    return text or ""


# ── 一站式：给一次会话/一轮对话算出全部路由 ─────────────────────────────────

class SessionRoutes:
    """一次会话的完整路由决策。构造后不再变（语言对由 session_config 锁定）。"""

    def __init__(self, src_lang: str, tgt_lang: str, mode: str = "interpreter",
                 hotwords: list[str] | None = None,
                 default_hotwords: dict | None = None,
                 use_llm_translator: bool = True,
                 use_llm_zh_to_sea: bool = False,
                 voice_role: str | None = None):
        self.mode = mode
        self.src = normalize(src_lang)
        self.tgt = normalize(tgt_lang)
        self.asr = route_asr(self.src, hotwords, default_hotwords)
        self.translate = route_translate(
            self.src, self.tgt, use_llm_translator, use_llm_zh_to_sea)
        self.voice_role = voice_role or DEFAULT_VOICE

    def describe(self) -> str:
        return (f"{display(self.src)}→{display(self.tgt)} "
                f"[{self.mode}] ASR={self.asr.backend} "
                f"MT={self.translate.backend} voice={self.voice_role}")

    def __repr__(self):
        return f"SessionRoutes({self.describe()})"

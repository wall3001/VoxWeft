"""LanguageRouter — pick a backend per (task, language pair). Pure functions.

Why it exists: no single open model covers every language we care about.
ASR has two paths (Qwen3-ASR for the languages it ships with, Whisper for the
rest), translation has two (an MLX chat LLM, a dedicated MT model), TTS has
one. Keeping the choices in one table beats if-else scattered over the
server; adding a language is one row.

Constraints, all measured:
  * Language codes arrive in every shape (lo / Lao / lao_Laoo / 老挝语).
    `normalize()` maps them to FLORES-200 before anything else looks at them.
  * zh→low-resource with a small chat LLM guesses the wrong language
    (Khmer/Thai script mixed into Lao). Route those through dedicated MT.
  * SEA↔SEA is direct, never via Chinese — two hops double latency and
    compound errors.
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
DEU = "deu_Latn"
FRA = "fra_Latn"
SPA = "spa_Latn"
ITA = "ita_Latn"
POR = "por_Latn"
RUS = "rus_Cyrl"
JPN = "jpn_Jpan"
KOR = "kor_Hang"
ARA = "arb_Arab"

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
    "de": DEU, "deu": DEU, "deu_latn": DEU, "german": DEU, "德语": DEU,
    "fr": FRA, "fra": FRA, "fra_latn": FRA, "french": FRA, "法语": FRA,
    "es": SPA, "spa": SPA, "spa_latn": SPA, "spanish": SPA, "西班牙语": SPA,
    "it": ITA, "ita": ITA, "ita_latn": ITA, "italian": ITA, "意大利语": ITA,
    "pt": POR, "por": POR, "por_latn": POR, "portuguese": POR, "葡萄牙语": POR,
    "ru": RUS, "rus": RUS, "rus_cyrl": RUS, "russian": RUS, "俄语": RUS,
    "ja": JPN, "jpn": JPN, "jpn_jpan": JPN, "japanese": JPN, "日语": JPN,
    "ko": KOR, "kor": KOR, "kor_hang": KOR, "korean": KOR, "韩语": KOR,
    "ar": ARA, "arb": ARA, "arb_arab": ARA, "arabic": ARA, "阿拉伯语": ARA,
}

# 向用户展示的名字（prompt 里用）
LANG_DISPLAY = {
    ZH: "中文", LAO: "老挝语", THA: "泰语", KHM: "高棉语", MYA: "缅甸语",
    VIE: "越南语", ENG: "英语", IND: "印尼语", MSA: "马来语",
    DEU: "德语", FRA: "法语", SPA: "西班牙语", ITA: "意大利语", POR: "葡萄牙语",
    RUS: "俄语", JPN: "日语", KOR: "韩语", ARA: "阿拉伯语",
}

# FLORES-200 → language name as Qwen3-ASR's `language=` expects (matched
# case-insensitively against config.support_languages by mlx-audio).
FLORES_TO_NAME = {
    ZH: "Chinese", LAO: "Lao", THA: "Thai", KHM: "Khmer",
    MYA: "Burmese", VIE: "Vietnamese", ENG: "English",
    IND: "Indonesian", MSA: "Malay", DEU: "German", FRA: "French",
    SPA: "Spanish", ITA: "Italian", POR: "Portuguese", RUS: "Russian",
    JPN: "Japanese", KOR: "Korean", ARA: "Arabic",
}

# Languages stock Qwen3-ASR (0.6B/1.7B) ships with, among the ones we route.
# Lao / Khmer / Burmese are not in its 30 — those go to Whisper-large-v3,
# which does have them. A fine-tune that adds them can extend this set.
QWEN3_ASR_LANGS = {ZH, VIE, THA, ENG, IND, MSA, DEU, FRA, SPA, ITA, POR, RUS, JPN, KOR, ARA}

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
    """Pick the ASR backend by source language.

    Qwen3-ASR for what it ships with (zh/vi/th/en/id/ms …), Whisper-large-v3
    for the low-resource SEA languages it lacks (lo/km/my). The `lang` field
    is what the backend's `language=` argument expects: a language name for
    Qwen3-ASR, an ISO-639-1 code for Whisper.
    """
    code = normalize(src_lang)
    hw = list(hotwords or [])
    if not hw and default_hotwords:
        hw = list(default_hotwords.get(code, []))

    if code in QWEN3_ASR_LANGS:
        return ASRRoute("qwen3asr", FLORES_TO_NAME[code], hw)
    return ASRRoute("whisper", whisper_lang(code), hw)


def whisper_lang(code: str) -> str:
    """Whisper takes ISO-639-1 two-letter codes, not FLORES."""
    return {
        ENG: "en", IND: "id", MSA: "ms", VIE: "vi",
        THA: "th", LAO: "lo", KHM: "km", MYA: "my", ZH: "zh",
        DEU: "de", FRA: "fr", SPA: "es", ITA: "it", POR: "pt", RUS: "ru",
        JPN: "ja", KOR: "ko", ARA: "ar",
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

    规则（都有实测依据）：
      * →老挝/高棉/缅甸：专用 MT（NLLB / Hy-MT2）。小 LLM 生成这些低资源文字会蒙错语言。
      * 其余任意语对：LLM **直译**，不经中文或英文中转（es→ms、de→vi 都是一跳）。

    `use_llm_zh_to_sea` 是现场逃生开关（VOXWEFT_USE_LLM_ZH_TO_SEA=1）：
    打开后连第一类也交给 LLM。默认关。
    """
    src, tgt = normalize(src_lang), normalize(tgt_lang)
    if src == tgt:
        return TranslateRoute("nllb", "same-lang (NLLB echo)")

    if not use_llm_translator:
        return TranslateRoute("nllb", "LLM translator disabled")

    # 低资源目标语（lo/km/my）：小 LLM 生成这些文字会蒙错语言（实测：
    # gemma3-4b 出高棉文、Qwen3-4B 出泰文混排+循环）。走专用 MT。
    # 判据只看**目标**语言：lo→zh 只要看懂老挝文、生成中文，4B 强项。
    if _prefers_nllb(tgt) and not use_llm_zh_to_sea:
        return TranslateRoute("nllb", f"->{tgt} (small LLM unreliable)")

    # 其余任意语对**直译**——es→ms、de→vi、ja→fr 都不经中文或英文中转。
    # 两跳会错误累积、延迟翻倍；4B 对这些高/中资源语言的直译实测可用。

    return TranslateRoute("llm", "direct, no pivot")


# ── TTS 路由 ────────────────────────────────────────────────────────────────
#
# One TTS backend. Voice selection is by reference audio (cloning) on the
# bridge, never by text: VoxCPM2 has no style-prefix syntax — anything in the
# text is spoken. `voice_role` is carried through the session so a host can
# map roles to reference voices; the runtime itself does not alter text.
VOICE_ROLES = ("interpreter", "agent_default", "agent_warm")
DEFAULT_VOICE = "interpreter"


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

    def voice_id(self, role: str | None = None) -> str:
        """Voice to ask the bridge for. Convention: `<tgt-iso>_tgt` for the
        interpreter (the voice that speaks the translation), `<tgt-iso>_agent`
        for dialogue; the bridge falls back to its default when the id is not
        installed, so this is a preference, not a requirement."""
        iso = whisper_lang(self.tgt)
        suffix = "agent" if (role or self.voice_role).startswith("agent") else "tgt"
        return f"{iso}_{suffix}"

    def describe(self) -> str:
        return (f"{display(self.src)}→{display(self.tgt)} "
                f"[{self.mode}] ASR={self.asr.backend} "
                f"MT={self.translate.backend} voice={self.voice_role}")

    def __repr__(self):
        return f"SessionRoutes({self.describe()})"

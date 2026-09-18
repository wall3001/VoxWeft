"""LanguageRouter 单元测试 —— 纯函数，不需要模型。

    pytest tests/test_runtime_router.py

重点验证三个实测踩过的坑不回归：
  1. 各种老挝语写法都必须归一成同一件事（不归一会静默退化成泰文）
  2. zh→lo 必须走 NLLB（小 LLM 会蒙错语言）
  3. SEA↔SEA 必须直译，不经中文中转
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from voxweft.runtime import router as LR


def test_normalize_all_lao_forms():
    """这是最重要的一个测试。任何一条漏网都会让老挝语静默变泰文。"""
    for form in ["lo", "Lao", "LAO", "lao_Laoo", "laos", "laotian", "LAO_LAOO",
                 "老挝语", "老挝"]:
        got = LR.normalize(form)
        assert got == "lao_Laoo", f"normalize({form!r}) = {got!r}"
    print("✓ test_normalize_all_lao_forms")


def test_normalize_all_langs():
    cases = {
        "zh": "zho_Hans", "中文": "zho_Hans", "zh-Hans": "zho_Hans",
        "th": "tha_Thai", "泰语": "tha_Thai",
        "km": "khm_Khmr", "高棉语": "khm_Khmr",
        "my": "mya_Mymr", "缅甸语": "mya_Mymr",
        "vi": "vie_Latn", "越南语": "vie_Latn",
        "en": "eng_Latn", "英语": "eng_Latn",
        "id": "ind_Latn", "印尼语": "ind_Latn",
        "ms": "zsm_Latn", "马来语": "zsm_Latn",
    }
    for src, want in cases.items():
        got = LR.normalize(src)
        assert got == want, f"normalize({src!r}) = {got!r}, want {want!r}"
    print("✓ test_normalize_all_langs")


def test_normalize_unknown_passes_through():
    """认不出就原样返回，让下游报错——不静默猜一个语言。"""
    assert LR.normalize("klingon") == "klingon"
    print("✓ test_normalize_unknown_passes_through")


# ── ASR 路由 ────────────────────────────────────────────────────────────────

def test_asr_qwen3asr_for_finetuned_langs():
    for code in ["lao_Laoo", "tha_Thai", "khm_Khmr", "mya_Mymr",
                 "vie_Latn", "zho_Hans"]:
        r = LR.route_asr(code)
        assert r.backend == "qwen3asr", f"{code} should hit SEA-finetuned Qwen3-ASR, got {r.backend}"
    print("✓ test_asr_qwen3asr_for_finetuned_langs")


def test_asr_whisper_for_others():
    for code in ["ind_Latn", "zsm_Latn", "eng_Latn"]:
        r = LR.route_asr(code)
        assert r.backend == "whisper", f"{code} should hit Whisper, got {r.backend}"
    print("✓ test_asr_whisper_for_others")


def test_asr_lang_string_is_mlx_label():
    """喂给 mlx 的必须是训练标签（小写），不是 FLORES 码。"""
    assert LR.route_asr("lao_Laoo").lang == "lao"
    assert LR.route_asr("khm_Khmr").lang == "khmer"
    assert LR.route_asr("zho_Hans").lang == "chinese"
    print("✓ test_asr_lang_string_is_mlx_label")


def test_asr_whisper_lang_is_iso():
    """Whisper 认两字母 ISO 码。"""
    assert LR.route_asr("ind_Latn").lang == "id"
    assert LR.route_asr("eng_Latn").lang == "en"
    print("✓ test_asr_whisper_lang_is_iso")


def test_asr_hotword_defaults():
    defaults = {"lao_Laoo": ["ວຽງຈັນ"]}
    r = LR.route_asr("lao_Laoo", default_hotwords=defaults)
    assert r.hotwords == ["ວຽງຈັນ"], f"got {r.hotwords}"
    # 客户端传了就用客户端的
    r2 = LR.route_asr("lao_Laoo", hotwords=["自定义"], default_hotwords=defaults)
    assert r2.hotwords == ["自定义"], f"got {r2.hotwords}"
    print("✓ test_asr_hotword_defaults")


# ── 翻译路由 ────────────────────────────────────────────────────────────────

def test_zh_to_sea_uses_nllb():
    """实测：小 LLM 在低资源语系上蒙错语言，必须 NLLB。"""
    r = LR.route_translate("zho_Hans", "lao_Laoo")
    assert r.backend == "nllb", f"zh→lo must use NLLB, got {r.backend}"
    print("✓ test_zh_to_sea_uses_nllb")


def test_zh_to_sea_llm_when_opted_in():
    r = LR.route_translate("zho_Hans", "lao_Laoo", use_llm_zh_to_sea=True)
    assert r.backend == "llm", f"opt-in should allow LLM, got {r.backend}"
    print("✓ test_zh_to_sea_llm_when_opted_in")


def test_sea_to_zh_uses_llm():
    """lo→zh 是 4B 的强项：完整句 267-310ms。"""
    r = LR.route_translate("lao_Laoo", "zho_Hans")
    assert r.backend == "llm", f"lo→zh should use LLM, got {r.backend}"
    print("✓ test_sea_to_zh_uses_llm")


def test_sea_to_sea_is_nllb_direct():
    """SEA↔SEA 不经中文中转——两跳错误累积 + 延迟翻倍。"""
    for s, t in [("lao_Laoo", "tha_Thai"), ("khm_Khmr", "vie_Latn"),
                 ("mya_Mymr", "khm_Khmr"), ("tha_Thai", "lao_Laoo")]:
        r = LR.route_translate(s, t)
        assert r.backend == "nllb", f"{s}→{t} must be NLLB direct, got {r.backend}"
        assert "direct" in r.reason
    print("✓ test_sea_to_sea_is_nllb_direct")


def test_llm_disabled_falls_to_nllb():
    r = LR.route_translate("lao_Laoo", "zho_Hans", use_llm_translator=False)
    assert r.backend == "nllb"
    print("✓ test_llm_disabled_falls_to_nllb")


# ── TTS 音色 ────────────────────────────────────────────────────────────────

def test_voice_caption_is_empty():
    """音色前缀必须为空 —— 实测结论，不是设计选择。

    VoxCPM2 会把 `(一位沉稳清晰的播音员)` 当正文念出来（ASR 回环验证：
    2.56s → 4.80s，识别结果里出现「播音员」）。所以任何角色都不能加前缀。
    """
    for role in ("interpreter", "agent_default", "agent_warm", None, "unknown"):
        got = LR.voice_caption(role)
        assert got == "", f"voice_caption({role!r}) 应返回空串，实际 {got!r}"
    print("\u2713 test_voice_caption_is_empty")


def test_apply_voice_is_identity():
    """apply_voice 不得改动文本（否则前缀会被念出来）。"""
    for text in ("欢迎参加东盟数字经济会议。", "ສະບາຍດີ", "a"):
        got = LR.apply_voice(text, "interpreter")
        assert got == text, f"apply_voice 改动了文本: {text!r} → {got!r}"
    assert LR.apply_voice("", "interpreter") == ""
    print("\u2713 test_apply_voice_is_identity")



# ── 会话级 ──────────────────────────────────────────────────────────────────

def test_session_routes_describe():
    sr = LR.SessionRoutes("lo", "zh", mode="interpreter")
    d = sr.describe()
    assert "老挝语→中文" in d, d
    assert "ASR=qwen3asr" in d, d
    assert "MT=llm" in d, d
    print("✓ test_session_routes_describe")


def test_session_routes_agent_mode():
    sr = LR.SessionRoutes("th", "zh", mode="agent", voice_role="agent_warm")
    assert sr.mode == "agent"
    assert sr.voice_role == "agent_warm"
    print("✓ test_session_routes_agent_mode")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"✗ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

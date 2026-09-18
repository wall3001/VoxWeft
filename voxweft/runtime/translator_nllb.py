"""NLLB-200 translator engine — zh ↔ SEA low-resource languages (lao/khm/mya/tha/vie/...).

Model: facebook/nllb-200-1.3B (seq2seq, 200 languages).
Language codes: FLORES-200 style, e.g. zho_Hans, lao_Laoo, khm_Khmr, mya_Mymr,
tha_Thai, vie_Latn, ind_Latn, zsm_Latn.
"""
import time
import logging

log = logging.getLogger("voxweft.translator_nllb")

# Short alias → FLORES-200 code
LANG_ALIASES = {
    "zh": "zho_Hans", "zh-Hans": "zho_Hans", "zh-Hant": "zho_Hant",
    "lo": "lao_Laoo", "lao": "lao_Laoo",
    "km": "khm_Khmr", "khmer": "khm_Khmr",
    "my": "mya_Mymr", "burmese": "mya_Mymr",
    "th": "tha_Thai", "thai": "tha_Thai",
    "vi": "vie_Latn", "viet": "vie_Latn",
    "id": "ind_Latn", "ms": "zsm_Latn",
    "en": "eng_Latn",
}


def resolve_lang(code: str) -> str:
    return LANG_ALIASES.get(code, code)


class NLLBTranslator:
    def __init__(self, model_dir: str, device: str = "cuda:0", num_beams: int = 4,
                 no_repeat_ngram: int = 4):
        self._model_dir = model_dir
        self._device = device
        self._num_beams = num_beams
        self._no_repeat_ngram = no_repeat_ngram
        self._model = None
        self._tok = None

    def load(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        t0 = time.perf_counter()
        self._tok = AutoTokenizer.from_pretrained(self._model_dir)
        # CUDA can run fp16; MPS/CPU on Apple Silicon prefers fp32
        dtype = torch.float16 if "cuda" in self._device else torch.float32
        self._model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                self._model_dir, torch_dtype=dtype
            )
            .to(self._device)
            .eval()
        )
        # warmup
        self.translate("你好。", "zho_Hans", "lao_Laoo")
        log.info(
            "NLLB loaded (%s, %s, %.1fs)",
            self._model_dir, self._device, time.perf_counter() - t0,
        )
        return self

    def translate(self, text: str, src_lang: str, tgt_lang: str,
                  max_new_tokens: int = 256) -> dict:
        """Translate text. Returns {'text': str, 'latency_ms': float}."""
        import torch

        t0 = time.perf_counter()
        src = resolve_lang(src_lang)
        tgt = resolve_lang(tgt_lang)

        self._tok.src_lang = src
        inputs = self._tok(text, return_tensors="pt", truncation=True,
                           max_length=512).to(self._device)
        bos_id = self._tok.convert_tokens_to_ids(tgt)
        with torch.inference_mode():
            ids = self._model.generate(
                **inputs,
                forced_bos_token_id=bos_id,
                max_new_tokens=max_new_tokens,
                num_beams=self._num_beams,
                # 中→老实测：不加这条，双分句的中文会把解码带进死循环，
                # 吐出「ແຕ່ວ່າ, ມັນຍັງບໍ່ທັນເປັນໄປໄດ້.」×2 这类跟原文无关的
                # 套话，或者同一个词连刷四十遍。禁掉 4-gram 重复后整条
                # 语料恢复正常，且 MPS 上 beam 数是带宽瓶颈不是算力瓶颈，
                # 4 beam 和 2 beam 实测同速（510ms vs 536ms/句）。
                no_repeat_ngram_size=self._no_repeat_ngram,
            )
        out = self._tok.batch_decode(ids, skip_special_tokens=True)[0]
        latency = (time.perf_counter() - t0) * 1000
        return {"text": out.strip(), "latency_ms": latency}

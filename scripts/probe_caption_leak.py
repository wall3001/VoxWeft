"""验证 VoxCPM2 是否会念出音色 caption 前缀。

背景：语言路由里给 TTS 文本加了 `(一位沉稳清晰的播音员)` 前缀，本意是
当「隐式音色条件」（当初的注释写的是「caption 字未被读出」）。用户实测
听到这句话被念出来了。

判定方法：合成两段——带前缀 / 不带前缀，各自用 ASR 回环识别。
如果带前缀那段的识别结果里出现「播音员」「沉稳」等词，说明确实被读了。
"""
import base64
import io
import sys

import numpy as np
import requests
import soundfile as sf

BRIDGE = "" + __import__("os").environ.get("VOXWEFT_BRIDGE_URL", "http://127.0.0.1:8133") + ""
BODY = "欢迎参加东盟数字经济会议。"
CAPTION = "(一位沉稳清晰的播音员)"


def tts(text: str) -> np.ndarray:
    r = requests.post(f"{BRIDGE}/generate", timeout=180,
                      json={"target_text": text, "stream": False,
                            "temperature": 1.0, "cfg_value": 2.0})
    r.raise_for_status()
    a, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a


def asr(a: np.ndarray, sr: int = 48000) -> str:
    n = int(len(a) * 16000 / sr)
    a16 = np.interp(np.linspace(0, len(a), n, endpoint=False),
                    np.arange(len(a)), a).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, (np.clip(a16, -1, 1) * 32767).astype(np.int16), 16000,
             format="WAV", subtype="PCM_16")
    url = "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()
    r = requests.post(f"{BRIDGE}/v1/chat/completions", timeout=300, json={
        "model": "mb-asr",
        "messages": [{"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": url}}]}],
        "audio_duration_sec": len(a16) / 16000,
        "language": "chinese"})
    r.raise_for_status()
    c = r.json()["choices"][0]["message"]["content"]
    _, sep, t = c.partition("<asr_text>")
    return (t if sep else c).strip()


def rms(a):
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else 0.0


print(f"正文：{BODY}")
print(f"前缀：{CAPTION}\n")

for label, text in (("不带前缀", BODY), ("带前缀", CAPTION + BODY)):
    a = tts(text)
    heard = asr(a)
    print(f"── {label} ──")
    print(f"  时长 {len(a)/48000:.2f}s   RMS {rms(a):.4f}")
    print(f"  ASR 回环：{heard}")
    leaked = any(k in heard for k in ("播音员", "沉稳", "清晰"))
    print(f"  → 前缀被念出：{'❌ 是' if leaked else '✅ 否'}")
    print()

"""Orchestrator 测试 —— 用假引擎，不需要模型/网络。

    pytest tests/test_runtime_orchestrator.py

验的是**编排正确性**：事件顺序、级联重叠、模式差异、打断行为。
模型的输出质量不在这里测（那是验收素材的事）。
"""
import asyncio
import sys
import time

import numpy as np

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
from voxweft.runtime.segmenter import SR, FRAME_SAMPLES
from voxweft.runtime.orchestrator import Orchestrator


# ── 假引擎 ──────────────────────────────────────────────────────────────────

class FakeASR:
    def __init__(self, text="测试文本", delay=0.0):
        self.text, self.delay, self.calls = text, delay, []
    def transcribe(self, audio, language=None, hotwords=None):
        self.calls.append({"lang": language, "samples": len(audio), "hw": hotwords})
        if self.delay:
            time.sleep(self.delay)
        return {"text": self.text, "latency_ms": self.delay * 1000}


class FakeTranslator:
    def __init__(self, delay=0.0):
        self.delay, self.calls = delay, []
    def translate(self, text, src, tgt):
        self.calls.append((text, src, tgt))
        if self.delay:
            time.sleep(self.delay)
        return {"text": f"[{tgt}]{text}", "latency_ms": self.delay * 1000}


class FakeTTS:
    def __init__(self, chunks=3, delay=0.0):
        self.chunks, self.delay, self.calls = chunks, delay, []
    def synthesize_stream(self, text):
        self.calls.append(text)
        for i in range(self.chunks):
            if self.delay:
                time.sleep(self.delay)
            yield np.zeros(480, dtype=np.float32)   # 10ms @48k


class FakeFiller:
    def get_filler(self):
        return "嗯，", b"\x00" * 100


class FakeLLM:
    def __init__(self, sentences=("第一句。", "第二句。")):
        self.sentences, self.calls = sentences, []
    def stream_sentences(self, user_text, *, system_prompt="", history=None):
        self.calls.append({"user": user_text, "sys": system_prompt,
                           "hist": list(history or [])})
        for i, s in enumerate(self.sentences):
            yield {"sentence": s, "ttfs_ms": 100.0 if i == 0 else 0.0}


def tone(n_frames, amp=0.2):
    t = np.arange(FRAME_SAMPLES * n_frames) / SR
    w = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return [w[i:i + FRAME_SAMPLES] for i in range(0, len(w), FRAME_SAMPLES)]


def silence(n_frames):
    return [np.zeros(FRAME_SAMPLES, dtype=np.float32) for _ in range(n_frames)]


class Harness:
    """收集下发的事件，按顺序。"""
    def __init__(self):
        self.events, self.audio_bytes = [], 0
    async def send_json(self, d):
        self.events.append(d)
    async def send_bytes(self, b):
        self.audio_bytes += len(b)
    def types(self):
        return [e["type"] for e in self.events]
    def first_index(self, t):
        ts = self.types()
        return ts.index(t) if t in ts else -1


def make(engines, mode="interpreter", **kw):
    h = Harness()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    orch = Orchestrator(
        mode=mode, src_lang="lao_Laoo", tgt_lang="zho_Hans",
        send_json=h.send_json, send_bytes=h.send_bytes,
        engines=engines, use_smart_turn=False, **kw)
    return orch, h, loop


def drive(orch, h, loop, frames, *, expect_turns=1):
    """喂音频 + 等尾段。

    结尾务必 aclose()：worker 是个常驻 task，不关掉的话事件循环销毁时会刷
    一屏 "Event loop is closed" / "Task was destroyed" —— 那些噪声会盖掉
    真正的断言失败信息。
    """
    async def go():
        for f in frames:
            orch.feed(f)
        await orch.flush()
        # 等 worker 把队列做完
        for _ in range(200):
            if orch._seg_queue.empty() and not orch._tasks:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        await orch.aclose()
    loop.run_until_complete(go())
    loop.close()


# ── 模式 A · 同传 ───────────────────────────────────────────────────────────

def test_interpreter_event_order():
    eng = {"asr": FakeASR("ສະບາຍດີ"), "translate": FakeTranslator(),
           "tts": FakeTTS()}
    orch, h, loop = make(eng)
    drive(orch, h, loop, tone(40) + silence(20))
    ts = h.types()
    for t in ["asr", "translation", "audio_start", "audio_bytes_marker", "turn_end"]:
        pass
    assert "asr" in ts, ts
    assert "translation" in ts, ts
    assert ts.index("asr") < ts.index("translation"), ts
    assert ts.index("translation") < h.first_index("audio_start"), ts
    assert "audio_end" in ts, ts
    assert "turn_end" in ts, ts
    assert h.audio_bytes > 0, "no audio emitted"
    print("✓ test_interpreter_event_order")


def test_interpreter_has_no_llm():
    """同传是转述，不该有 LLM 事件。"""
    eng = {"asr": FakeASR(), "translate": FakeTranslator(), "tts": FakeTTS(),
           "llm": FakeLLM()}
    orch, h, loop = make(eng)
    drive(orch, h, loop, tone(40) + silence(20))
    assert "llm_sentence" not in h.types(), "interpreter must not use LLM"
    print("✓ test_interpreter_has_no_llm")


def test_interpreter_two_segments():
    eng = {"asr": FakeASR(), "translate": FakeTranslator(), "tts": FakeTTS()}
    orch, h, loop = make(eng)
    drive(orch, h, loop, tone(40) + silence(20) + tone(40) + silence(20))
    n = sum(1 for e in h.events if e["type"] == "translation")
    assert n == 2, f"expected 2 translations, got {n}"
    assert h.types().count("segment_committed") == 2
    print("✓ test_interpreter_two_segments")


def test_interpreter_uses_router_lang():
    """ASR 拿到的必须是路由后的语言串（后端期望的形式）。"""
    asr = FakeASR()
    eng = {"asr": asr, "translate": FakeTranslator(), "tts": FakeTTS()}
    h = Harness()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    orch = Orchestrator(
        mode="interpreter", src_lang="vi", tgt_lang="zh",
        send_json=h.send_json, send_bytes=h.send_bytes,
        engines=eng, use_smart_turn=False)
    drive(orch, h, loop, tone(40) + silence(20))
    assert asr.calls, "ASR never called"
    assert asr.calls[0]["lang"] == "Vietnamese", f"got {asr.calls[0]['lang']!r}"
    print("✓ test_interpreter_uses_router_lang")


def test_interpreter_speaks_translation_verbatim():
    """送进 TTS 的必须是译文**原样**，一个字都不多。

    TTS 是哑巴 render，给什么念什么——任何控制标记混进文本都会被当着听众
    念出来（DEVLOG §2.2）。所以 runtime 不允许改动合成文本。
    """
    tts = FakeTTS()
    eng = {"asr": FakeASR(), "translate": FakeTranslator(), "tts": tts}
    orch, h, loop = make(eng)
    drive(orch, h, loop, tone(40) + silence(20))
    assert tts.calls, "TTS never called"
    spoken, shown = tts.calls[0], h.events[h.first_index("translation")]["text"]
    assert spoken == shown, f"念的和屏幕上的不一致：念 {spoken!r} / 显示 {shown!r}"
    print("✓ test_interpreter_speaks_translation_verbatim")


# ── 模式 B · 对话 ───────────────────────────────────────────────────────────

def test_agent_uses_llm_and_tts_per_sentence():
    llm = FakeLLM(("你好呀。", "很高兴认识你。"))
    tts = FakeTTS()
    eng = {"asr": FakeASR("你好"), "translate": FakeTranslator(), "tts": tts,
           "llm": llm}
    orch, h, loop = make(eng, mode="agent", system_prompt="你是助手")
    drive(orch, h, loop, tone(40) + silence(20))
    ts = h.types()
    assert "llm_sentence" in ts, ts
    # 两句 → 两次 TTS 调用
    assert len(tts.calls) == 2, f"expected 2 TTS calls, got {len(tts.calls)}"
    assert ts.count("audio_start") == 2, f"got {ts.count('audio_start')}"
    # 系统 prompt 传到了 LLM
    assert llm.calls[0]["sys"] == "你是助手", llm.calls[0]
    print("✓ test_agent_uses_llm_and_tts_per_sentence")


def test_agent_state_transitions():
    eng = {"asr": FakeASR("你好"), "translate": FakeTranslator(), "tts": FakeTTS(),
           "llm": FakeLLM()}
    orch, h, loop = make(eng, mode="agent")
    drive(orch, h, loop, tone(40) + silence(20))
    states = [e["state"] for e in h.events if e["type"] == "state"]
    assert "thinking" in states and "speaking" in states and "idle" in states, states
    assert states.index("thinking") < states.index("speaking") < states.index("idle")
    print("✓ test_agent_state_transitions")


def test_agent_history_accumulates():
    eng = {"asr": FakeASR("你好"), "translate": FakeTranslator(), "tts": FakeTTS(),
           "llm": FakeLLM(("好的。",))}
    orch, h, loop = make(eng, mode="agent")
    drive(orch, h, loop, tone(40) + silence(20))
    assert len(orch._history) == 2, f"got {orch._history}"
    assert orch._history[0]["role"] == "user"
    assert orch._history[1]["role"] == "assistant"
    print("✓ test_agent_history_accumulates")


def test_agent_filler_sent():
    eng = {"asr": FakeASR("你好"), "translate": FakeTranslator(), "tts": FakeTTS(),
           "llm": FakeLLM(), "filler": FakeFiller()}
    orch, h, loop = make(eng, mode="agent")
    drive(orch, h, loop, tone(40) + silence(20))
    assert "filler" in h.types(), h.types()
    print("✓ test_agent_filler_sent")


# ── 打断 ────────────────────────────────────────────────────────────────────

def _pending_segments(orch):
    """队列里还没被 worker 取走的「段」数量（None 哨兵不算）。"""
    n = 0
    while True:
        try:
            item = orch._seg_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if item is not None:
            n += 1
    return n


def test_interrupt_emits_event_and_clears():
    eng = {"asr": FakeASR(), "translate": FakeTranslator(), "tts": FakeTTS()}
    orch, h, loop = make(eng)

    async def go():
        # 先塞两段进队列，模拟「积压了但还没做完」
        from voxweft.runtime.segmenter import CommittedSegment
        orch._seg_queue.put_nowait((101, CommittedSegment(
            np.zeros(SR, dtype=np.float32), "endpoint")))
        orch._seg_queue.put_nowait((102, CommittedSegment(
            np.zeros(SR, dtype=np.float32), "endpoint")))
        await orch.interrupt("test")
        assert _pending_segments(orch) == 0, "queued segments must be dropped"
        await orch.aclose()

    loop.run_until_complete(go())
    loop.close()
    assert "interrupted" in h.types(), h.types()
    print("✓ test_interrupt_emits_event_and_clears")


def test_interrupt_cancels_inflight_tts():
    """打断后不该再有音频发出去。"""
    # 慢 TTS，好让打断插进来。20 块 × 50ms = 1s 总时长；打断在 0.2s 处。
    # 以前是 10ms/块 + 60ms 后打断：在快机器上（Linux 服务器）20 块在
    # 打断前就全发完了，测试变成在测机器速度。
    tts = FakeTTS(chunks=20, delay=0.05)
    eng = {"asr": FakeASR(), "translate": FakeTranslator(), "tts": tts}
    orch, h, loop = make(eng)

    async def go():
        for f in tone(40) + silence(20):
            orch.feed(f)
        await orch.flush()
        await asyncio.sleep(0.2)           # 让它开始合成（约 3-4 块）
        await orch.interrupt("barge")
        await asyncio.sleep(0.3)           # 给残余发送一点时间
        await orch.aclose()
        return h.audio_bytes

    n = loop.run_until_complete(go())
    loop.close()
    # 打断发生在 20 块之中——不可能全发完
    assert n < 20 * 480 * 2, f"interrupt did not stop TTS (sent {n} bytes)"
    print("✓ test_interrupt_cancels_inflight_tts")


# ── 路由集成 ────────────────────────────────────────────────────────────────

def test_zh_to_lo_routes_to_nllb_path():
    """zh→lo 必须走 NLLB（小 LLM 会蒙错语言）。"""
    h = Harness()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    eng = {"asr": FakeASR(), "translate": FakeTranslator(), "tts": FakeTTS()}
    orch = Orchestrator(
        mode="interpreter", src_lang="zh", tgt_lang="lo",
        send_json=h.send_json, send_bytes=h.send_bytes,
        engines=eng, use_smart_turn=False)
    assert orch.routes.translate.backend == "nllb", orch.routes.translate
    loop.run_until_complete(orch.aclose()); loop.close()
    print("✓ test_zh_to_lo_routes_to_nllb_path")


def test_sea_to_sea_direct():
    h = Harness()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    eng = {"asr": FakeASR(), "translate": FakeTranslator(), "tts": FakeTTS()}
    orch = Orchestrator(
        mode="interpreter", src_lang="lo", tgt_lang="th",
        send_json=h.send_json, send_bytes=h.send_bytes,
        engines=eng, use_smart_turn=False)
    assert orch.routes.translate.backend == "nllb", orch.routes.translate
    assert "direct" in orch.routes.translate.reason
    loop.run_until_complete(orch.aclose()); loop.close()
    print("✓ test_sea_to_sea_direct")


def test_whisper_route_for_lao():
    """老挝语走 Whisper：开源 Qwen3-ASR 不带 lo/km/my。"""
    h = Harness()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    wh = FakeASR("ສະບາຍດີ")
    eng = {"asr": FakeASR(), "whisper": wh, "translate": FakeTranslator(),
           "tts": FakeTTS()}
    orch = Orchestrator(
        mode="interpreter", src_lang="lo", tgt_lang="zh",
        send_json=h.send_json, send_bytes=h.send_bytes,
        engines=eng, use_smart_turn=False)
    drive(orch, h, loop, tone(40) + silence(20))
    assert wh.calls, "Whisper should have been called for Lao"
    assert wh.calls[0]["lang"] == "lo", wh.calls[0]
    print("✓ test_whisper_route_for_lao")


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
            import traceback
            print(f"✗ {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

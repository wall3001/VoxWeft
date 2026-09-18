"""IncrementalSegmenter 的单元测试 —— 不需要模型、不需要网络。

    cd crossweft/m3 && python3 -m pytest test_segmenter.py -v
    （或直接 python3 test_segmenter.py）

用合成音频验证切分行为：静音间隔、短段丢弃、长句强切、flush 收尾。
SmartTurn 用假实现代替，测的是「判定结果如何影响切分」这个逻辑本身。
"""
import asyncio
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), ".."))
from voxweft.runtime.segmenter import (
    IncrementalSegmenter, SR, FRAME_SAMPLES, FRAME_S)


def tone(n_frames: int, amp: float = 0.2) -> list[np.ndarray]:
    """n_frames 帧的「语音」——正弦波，RMS 高于阈值。"""
    t = np.arange(FRAME_SAMPLES * n_frames) / SR
    wave = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return [wave[i:i + FRAME_SAMPLES] for i in range(0, len(wave), FRAME_SAMPLES)]


def silence(n_frames: int) -> list[np.ndarray]:
    return [np.zeros(FRAME_SAMPLES, dtype=np.float32) for _ in range(n_frames)]


class FakeSmartTurn:
    """按预设脚本回答「说完了吗」——第 N 次调用返回第 N 个答案。"""
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0
    def should_endpoint(self, buf, is_speech_active=False):
        self.calls += 1
        if not self.answers:
            return True
        return self.answers.pop(0)


def run(frames, seg, *, settle=True):
    """喂帧。

    必须在事件循环里喂 —— 生产环境就是这样（feed 由 WS 回调调用，回调跑在
    loop 上）。段切分器的句尾判定用 get_running_loop() 起任务；在 loop 外喂
    帧会走「无运行循环 → 退化为纯静音判据」的分支，那样就测不到 SmartTurn
    路径了。
    """
    async def go():
        for f in frames:
            seg.feed(f)
        if settle:
            await asyncio.sleep(0.05)   # 让判定任务有机会执行
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(go())
    finally:
        loop.close()


# ── 纯静音判据（SmartTurn 关闭）──────────────────────────────────────────────

def test_single_segment():
    got = []
    seg = IncrementalSegmenter(got.append, use_smart_turn=False)
    run(tone(50) + silence(20), seg)     # 1.6s 说话 + 0.64s 静音
    assert len(got) == 1, f"expected 1 segment, got {len(got)}"
    assert got[0].duration_s > 1.5
    assert got[0].reason == "endpoint"
    print("✓ test_single_segment")


def test_two_segments():
    got = []
    seg = IncrementalSegmenter(got.append, use_smart_turn=False)
    run(tone(40) + silence(20) + tone(40) + silence(20), seg)
    assert len(got) == 2, f"expected 2 segments, got {len(got)}"
    print("✓ test_two_segments")


def test_short_blip_discarded():
    """咳一声（<0.6s）不该送去 ASR。"""
    got = []
    seg = IncrementalSegmenter(got.append, use_smart_turn=False)
    run(tone(5) + silence(20), seg)      # 0.16s 语音
    assert len(got) == 0, f"short blip should be discarded, got {len(got)}"
    print("✓ test_short_blip_discarded")


def test_breath_pause_not_split():
    """说话中的短停顿（<0.45s）不该切开——这是碎片的根源。"""
    got = []
    seg = IncrementalSegmenter(got.append, use_smart_turn=False)
    run(tone(30) + silence(8) + tone(30) + silence(20), seg)   # 中间 0.26s 停顿
    assert len(got) == 1, f"breath pause must not split, got {len(got)}"
    print("✓ test_breath_pause_not_split")


# ── SmartTurn 判据 ──────────────────────────────────────────────────────────

def test_smart_turn_blocks_premature_split():
    """静音够长但 SmartTurn 说「没说完」→ 不提交。"""
    got = []
    st = FakeSmartTurn([False, True])   # 第一次说不完整，第二次说完整
    seg = IncrementalSegmenter(got.append, smart_turn=st, use_smart_turn=True)
    run(tone(30) + silence(20), seg)    # 静音触发一次判定 → False
    assert len(got) == 0, "premature silence must not commit when SmartTurn says incomplete"
    run(tone(30) + silence(20), seg)    # 再一段 → 这次 True
    assert len(got) == 1, f"expected commit on True, got {len(got)}"
    print("✓ test_smart_turn_blocks_premature_split")


def test_smart_turn_failure_falls_back():
    """SmartTurn 抛异常 → 退化为纯静音判据（不比现状差）。"""
    class Broken:
        def should_endpoint(self, buf, is_speech_active=False):
            raise RuntimeError("onnx exploded")
    got = []
    seg = IncrementalSegmenter(got.append, smart_turn=Broken(), use_smart_turn=True)
    run(tone(40) + silence(20), seg)
    assert len(got) == 1, f"failure must degrade to silence rule, got {len(got)}"
    print("✓ test_smart_turn_failure_falls_back")


def test_speech_resumes_cancels_pending():
    """静音触发判定后用户又说了 → 这次判定作废，不提交。"""
    got = []
    st = FakeSmartTurn([True])          # 即使判定说「完了」
    seg = IncrementalSegmenter(got.append, smart_turn=st, use_smart_turn=True)

    async def go():
        for f in tone(20):
            seg.feed(f)
        for f in silence(15):           # 触发判定
            seg.feed(f)
        for f in tone(20):              # 用户又说了 → 作废
            seg.feed(f)
        await asyncio.sleep(0.05)       # 给判定任务机会执行
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(go())
    finally:
        loop.close()
    assert len(got) == 0, f"resumed speech must cancel pending commit, got {len(got)}"
    print("✓ test_speech_resumes_cancels_pending")


# ── 超长句强制切 ────────────────────────────────────────────────────────────

def test_forced_split_on_long_utterance():
    """长篇大论不能一直不出译文。"""
    got = []
    seg = IncrementalSegmenter(got.append, use_smart_turn=False,
                               max_utterance_s=3.0, max_seg_s=4.0)
    run(tone(int(3.5 / FRAME_S)) + silence(20), seg)   # 3.5s 连续说话
    assert len(got) >= 1, f"long utterance must be force-split, got {len(got)}"
    assert got[0].reason == "forced", f"expected forced, got {got[0].reason}"
    print("✓ test_forced_split_on_long_utterance")


def test_max_seg_hard_cap():
    """单段硬上限：即使一直在说也要切。"""
    got = []
    seg = IncrementalSegmenter(got.append, use_smart_turn=False,
                               max_utterance_s=99.0, max_seg_s=2.0)
    run(tone(int(2.5 / FRAME_S)), seg)
    assert len(got) >= 1, f"hard cap must trigger, got {len(got)}"
    print("✓ test_max_seg_hard_cap")


# ── flush ───────────────────────────────────────────────────────────────────

def test_flush_tail():
    """ptt_end 时残留音频不能丢。"""
    got = []
    seg = IncrementalSegmenter(got.append, use_smart_turn=False)
    run(tone(30), seg)                  # 说话，没等到静音
    assert len(got) == 0
    asyncio.new_event_loop().run_until_complete(seg.flush())
    assert len(got) == 1, f"flush must emit the tail, got {len(got)}"
    assert got[0].reason == "flush", f"reason should be flush, got {got[0].reason}"
    print("✓ test_flush_tail")


def test_flush_empty_is_noop():
    got = []
    seg = IncrementalSegmenter(got.append, use_smart_turn=False)
    asyncio.new_event_loop().run_until_complete(seg.flush())
    assert len(got) == 0
    print("✓ test_flush_empty_is_noop")


def test_reset_clears():
    got = []
    seg = IncrementalSegmenter(got.append, use_smart_turn=False)
    run(tone(30), seg)
    seg.reset()
    asyncio.new_event_loop().run_until_complete(seg.flush())
    assert len(got) == 0, "reset must drop buffered audio"
    print("✓ test_reset_clears")


if __name__ == "__main__":
    async def main():
        await asyncio.sleep(0)   # 让下面用到的 loop 就绪
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

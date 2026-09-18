"""IncrementalSegmenter — 边说边切段，不等 ptt_end。

现有的 ws_server 在同传上是**批量**的：FireRedVAD 开着时 `_commit_seg()`
是 no-op，音频一路攒到 ptt_end 才 detect_full 切一遍。结果是用户松手之前
一个字都出不来——满足不了「同传」。

这个类把「攒到松手」换成「边说边判定」：

    帧(32ms) ──▶ 累积 ──▶ 静音达阈值？──▶ 句尾判定 ──▶ 提交整段
                  │                          │
                  │                     三级过滤：
                  │                     ① RMS 静音 ≥ min_silence
                  │                     ② SmartTurn 判「真说完了」
                  │                     ③ 段长 ≥ min_seg
                  │
                  └─▶ 连续说话超 max_utterance ──▶ 强制提交（防长段不出译文）

为什么端点判定是异步的：SmartTurn 的 ONNX 推理 ~10ms，但**不能放在热路径**。
`_feed_frame` 每 32ms 被调一次，同步阻塞会拖垮采集。所以这里只在静音达标时
**起一个 async 判定任务**，判定期间音频继续缓冲、继续累积——判定结果回来时
若静音已恢复（用户又说了），直接丢弃这次判定，不影响数据。

SmartTurn 不可用或判定异常时，**退化为纯静音判据**（即当前行为），
保证任何情况下都不会比现在差。`smart_turn` 开关可一键回滚。
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("voxweft.segmenter")

SR = 16000
FRAME_S = 0.032          # 32ms 帧，与 ws_server FRAME_S 一致
FRAME_SAMPLES = int(SR * FRAME_S)   # 512


class CommittedSegment:
    """一次提交的语音段。seq 由调用方分配（服务端要保证全局单调）。"""

    __slots__ = ("audio", "duration_s", "reason", "t_commit")

    def __init__(self, audio: np.ndarray, reason: str):
        self.audio = audio
        self.duration_s = len(audio) / SR
        self.reason = reason          # "endpoint" | "forced" | "flush"
        self.t_commit = time.monotonic()

    def __repr__(self):
        return f"CommittedSegment({self.duration_s:.2f}s, {self.reason})"


class IncrementalSegmenter:
    """在线切分器。每帧喂进来，需要提交时通过 on_segment 回调吐出整段。

    用法::

        seg = IncrementalSegmenter(on_segment=handle)
        for frame in audio_frames:
            seg.feed(frame)          # 同步、极快
        await seg.flush()            # ptt_end：把尾段冲出来
    """

    def __init__(
        self,
        on_segment: Callable[[CommittedSegment], None],
        *,
        sil_thresh: float = 0.035,
        min_silence_s: float = 0.45,
        min_seg_s: float = 0.60,
        max_utterance_s: float = 15.0,
        max_seg_s: float = 20.0,
        smart_turn=None,
        use_smart_turn: bool = True,
        keep_trailing_frames: int = 6,
        turn_window_s: float = 4.0,
        recheck_interval_s: float = 0.4,
        max_carry_s: float = 2.0,
        long_min_silence_s: float = 0.12,
        hard_extra_s: float = 3.0,
        snap_s: float = 2.0,
        adaptive: bool = False,
        noise_ratio: float = 5.0,
        thresh_floor: float = 0.006,
    ):
        """
        sil_thresh:        RMS 低于此视为静音（沿用 ws_server 的老挝语实测值）
        min_silence_s:     静音持续多久才考虑提交（0.45s，同传可调小）
        min_seg_s:         短于此的段直接丢（咳嗽、嗯）
        max_utterance_s:   连续说话超此时长强制提交，防止长篇不出译文。
                           **别调太小**：它是「兜底」，不是主判据。展厅素材
                           实测——若它小于句尾判定的实际耗时，缓冲会先撞上它
                           而被强切，SmartTurn 永远等不到机会（结果是一串
                           整齐的 8s 段，恰好等于这个值，就是被它切的）。
        max_seg_s:         单段硬上限，超了切一刀（防内存无限涨）
        smart_turn:        SmartTurnDetector 实例，None 则纯静音判定
        use_smart_turn:    env 门控，默认 on
        keep_trailing_frames: 提交时保留的尾部静音帧数（给 ASR 收尾上下文）
        turn_window_s:     句尾判定喂给 SmartTurn 的尾部窗口长度。别设 <=0
                           （那会把整段累积喂进去，实测几乎永远 INCOMPLETE）
        max_carry_s:       短段顺延缓冲的上限。见 _commit 的短段分支。
        """
        self._on_segment = on_segment
        self._sil_thresh = sil_thresh
        self._min_silence_frames = max(1, int(min_silence_s / FRAME_S))
        self._min_seg_s = min_seg_s
        self._max_utterance_frames = max(1, int(max_utterance_s / FRAME_S))
        self._max_seg_samples = int(max_seg_s * SR)
        self._smart_turn = smart_turn
        self._use_smart_turn = use_smart_turn and smart_turn is not None
        self._keep_trailing = keep_trailing_frames
        self._turn_window_s = turn_window_s
        self._recheck_interval_s = recheck_interval_s
        self._max_carry_samples = int(max_carry_s * SR)
        # 超长句的「软强切」：不再见到一个 32ms 低能量帧就切（那多半是音节间的
        # 凹陷，刀口落在词中间，两侧的词都废），而是要求 ≥long_min_silence_s 的
        # 连续静音——词边界的停顿量级。实测（M2 16GB，vi/lo 真人朗读）：原策略让
        # 商用 ASR 的 CER 从 4.7% 涨到 16.3%，开源 ASR 从 10% 涨到 21%。
        self._long_min_silence_frames = max(1, int(long_min_silence_s / FRAME_S))
        # 连这点停顿都等不到（连读播音），到 max_utterance + hard_extra 时在最近
        # snap_s 秒里挑能量最低的一帧下刀，刀口之后的帧留给下一段。
        self._hard_samples = int((max_utterance_s + hard_extra_s) * SR)
        self._snap_frames = max(2, int(snap_s / FRAME_S))
        self._recheck_task = None
        self._dead = False
        # 自适应门限：跟踪噪声底（最小值统计，慢速回升），门限 = 噪声底 × noise_ratio，
        # 夹在 [thresh_floor, sil_thresh]。固定 0.035 是给展台近讲麦调的：手机录音
        # 中位帧 RMS 只有 0.02，70% 的语音帧被当静音（M2 实测），轻声起头的词
        # 直接被丢。
        self._adaptive = adaptive
        self._noise_ratio = noise_ratio
        self._thresh_floor = thresh_floor
        self._thresh_cap = sil_thresh
        self._noise_floor = None

        self._buf: list[np.ndarray] = []
        # 上一段太短、被退回来等着搭下一班车的音频。见 _commit 的短段分支。
        self._carry: list[np.ndarray] = []
        self._carry_speech = 0
        # 滚动语音环——只装最近 turn_window_s 秒的**语音**帧，喂 SmartTurn 用。
        # 与 _buf 解耦：_buf 送 ASR（要完整段），语音环喂韵律判定（只看句尾）。
        self._speech_ring: list[np.ndarray] = []
        self._sil_run = 0
        self._speech_samples = 0     # 本段内非静音样本数（用于 min_seg 判定）
        self._in_speech = False
        self._pending_turn_check: Optional[asyncio.Task] = None
        self._turn_check_gen = 0     # 判定任务代际，用于丢弃过期结果

    # ── 热路径 ──────────────────────────────────────────────────────────────

    def _update_thresh(self, rms: float) -> None:
        # 最小值统计：比噪声底更低就立刻跟下去，否则每帧回升 0.5%（约 6s 翻倍），
        # 这样长时间连续说话不会把「噪声底」抬成语音电平。
        if self._noise_floor is None:
            self._noise_floor = max(rms, 1e-4)
        elif rms < self._noise_floor:
            self._noise_floor = max(rms, 1e-4)
        else:
            self._noise_floor *= 1.005
        self._sil_thresh = min(self._thresh_cap, max(self._thresh_floor, self._noise_floor * self._noise_ratio))

    def feed(self, frame: np.ndarray) -> None:
        """喂一帧（512 float32 @16k）。必须快——每 32ms 调一次。"""
        rms = float(np.sqrt(np.mean(frame ** 2))) if len(frame) else 0.0
        if self._adaptive:
            self._update_thresh(rms)
        self._buf.append(frame)

        if rms > self._sil_thresh:
            # 有声音：清静音计数，作废未完的句尾判定（用户又说了）
            self._in_speech = True
            self._sil_run = 0
            self._speech_samples += len(frame)
            self._push_speech(frame)
            self._cancel_turn_check()
            # 硬上限：单段太长就切一刀，不无限攒。切在最近一段里最安静的帧上。
            total = self._total_samples()
            if total >= self._max_seg_samples or total >= self._hard_samples:
                self._commit_snapped()
            return

        # ── 静音帧 ──
        if not self._in_speech:
            # 还没开口：丢掉前置静音，只留少量作为起始上下文
            if len(self._buf) > self._keep_trailing:
                self._buf = self._buf[-self._keep_trailing:]
            return

        self._sil_run += 1

        # 已经连续说话很久了吗？（用整段时长判断，不看静音计数）
        long_utterance = self._total_samples() >= self._max_utterance_frames * FRAME_SAMPLES

        if self._sil_run < self._min_silence_frames:
            # 静音还不够长。超长句不等完整静音判据，但也不能见一帧就切——
            # 要一个词边界量级的停顿（long_min_silence）。判定已在跑时要等它。
            if (long_utterance and self._sil_run >= self._long_min_silence_frames
                    and not self._turn_check_pending()):
                self._commit("forced")
            return

        # 静音够长了 —— 该不该提交，看句尾判定
        if not self._use_smart_turn:
            self._commit("forced" if long_utterance else "endpoint")
        else:
            self._schedule_turn_check()

    def _push_speech(self, frame: np.ndarray) -> None:
        """维护滚动语音环，保留最近 turn_window_s 秒。"""
        self._speech_ring.append(frame)
        limit = int(self._turn_window_s * SR) if self._turn_window_s > 0 else 0
        if limit <= 0:
            return
        total = sum(len(f) for f in self._speech_ring)
        while len(self._speech_ring) > 1 and total - len(self._speech_ring[0]) >= limit:
            total -= len(self._speech_ring.pop(0))

    def _turn_check_pending(self) -> bool:
        return (self._pending_turn_check is not None
                and not self._pending_turn_check.done())

    # ── 句尾判定（异步，不阻塞热路径）────────────────────────────────────────

    def _schedule_turn_check(self) -> None:
        if self._turn_check_pending():
            return  # 已有判定在跑

        # 必须显式拿 running loop，不能用 asyncio.ensure_future ——
        # 后者的语义是「取当前事件循环」，在 feed() 被事件循环内的回调调用时
        # 恰好能拿到，但在事件循环外（脚本、线程、生命周期早期）会静默把任务
        # 挂到一个错误/已关闭的循环上：任务永不执行，判定永不发生，也不报错。
        # 这个 bug 的表现是「切分器只走强制切分，SmartTurn 一次都没被调用」。
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            log.debug("no running loop — turn check skipped, silence rule applies")
            self._commit("endpoint")
            return

        self._turn_check_gen += 1
        gen = self._turn_check_gen
        window = self._turn_window()
        self._pending_turn_check = loop.create_task(self._turn_check(window, gen))

    def _turn_window(self) -> list[np.ndarray]:
        """给句尾判定看的音频：**最近一段有语音的音频**。

        SmartTurn 是韵律模型，看句尾的升降调。这里有两个都实测踩过的坑：

        1. 喂整段累积缓冲（15s，含前面若干完整句）→ 几乎总判 INCOMPLETE
           （26 次调用 0 次 COMPLETE），永远等不到提交。
        2. 改成「取 _buf 尾部 N 秒」也错——触发判定时静音已持续 0.45s，
           缓冲尾部全是静音帧，取出来只有 96ms 静音，同样判不出来。

        所以要维护一个独立的滚动语音环（_speech_ring），只装最近的语音帧，
        与「累积缓冲」解耦：缓冲负责送 ASR，语音环负责喂韵律判定。
        """
        if self._turn_window_s <= 0:
            return list(self._speech_ring)
        need = int(self._turn_window_s * SR)
        take: list[np.ndarray] = []
        total = 0
        for f in reversed(self._speech_ring):
            take.append(f)
            total += len(f)
            if total >= need:
                break
        take.reverse()
        return take

    async def _turn_check(self, snapshot: list[np.ndarray], gen: int) -> None:
        """在线程池里跑 SmartTurn，不卡事件循环。"""
        try:
            loop = asyncio.get_running_loop()
            done = await loop.run_in_executor(None, self._smart_turn_done, snapshot)
        except Exception as e:
            log.warning("smart-turn check failed, falling back to silence: %s", e)
            done = True   # 判定失败 → 退化为纯静音判据，即当前行为

        if gen != self._turn_check_gen:
            return        # 期间用户又说话了，这次判定作废
        if not self._in_speech:
            return        # 段已经被别处提交了

        if done:
            self._commit("endpoint")
            return

        # 判定说「没说完」。**不能就这么等下一个静音帧** —— 静音已经在那儿
        # 了，_sil_run 只会继续涨，feed() 每帧都会再调 _schedule_turn_check()，
        # 而 _turn_check_pending() 恒为真 → 永远不再触发，段只能等 max_utterance
        # 强切。实测表现就是一串整齐的 15s 段。
        # 所以这里主动安排下一次判定，间隔 _recheck_interval_s。
        self._recheck_task = asyncio.get_running_loop().call_later(
            self._recheck_interval_s, self._recheck)

    def _recheck(self) -> None:
        """判定说不完整后，隔一会儿再问一次——用户可能只是换气。"""
        if not self._in_speech or self._dead:
            return
        if self._sil_run < self._min_silence_frames:
            return                      # 又说起话了，静音判据会接管
        self._schedule_turn_check()

    def _smart_turn_done(self, snapshot: list[np.ndarray]) -> bool:
        if self._smart_turn is None:
            return True
        return bool(self._smart_turn.should_endpoint(snapshot, is_speech_active=False))

    def _cancel_turn_check(self) -> None:
        """作废进行中的句尾判定。

        只递增代际号，**不调 Task.cancel()**：判定任务本来就靠代际号自我作废
        （见 _turn_check 开头的 gen 比较），硬取消反而有副作用——取消一个
        尚未开始执行的协程在 CPython 里表现为「已 done 但从未运行」，之后
        `_pending_turn_check.done()` 为真、`_schedule_turn_check` 会误判
        「没有在跑的判定」，加上 cancel 的竞态，判定会被静默丢弃。
        """
        self._turn_check_gen += 1
        self._pending_turn_check = None
        if self._recheck_task is not None:
            self._recheck_task.cancel()
            self._recheck_task = None

    # ── 提交 ────────────────────────────────────────────────────────────────

    def _total_samples(self) -> int:
        return sum(len(f) for f in self._buf)

    def _commit_snapped(self) -> None:
        """在缓冲尾部 snap 窗口内找能量最低的帧，切在那里；之后的帧重新喂回去。"""
        n = len(self._buf)
        if n <= self._snap_frames + 2:
            self._commit("forced")
            return
        lo = n - self._snap_frames
        best, best_rms = n, float("inf")
        for k in range(lo, n - 1):
            f = self._buf[k]
            r = float(np.sqrt(np.mean(f * f))) if len(f) else 0.0
            if r < best_rms:
                best_rms, best = r, k
        tail = self._buf[best:]
        self._buf = self._buf[:best]
        # speech_samples 要扣掉尾巴里的语音，_commit 用它判 min_seg
        tail_speech = sum(len(f) for f in tail
                          if len(f) and float(np.sqrt(np.mean(f * f))) > self._sil_thresh)
        self._speech_samples = max(0, self._speech_samples - tail_speech)
        self._commit("forced")
        for f in tail:
            self.feed(f)

    def _commit(self, reason: str) -> None:
        """打包当前缓冲为一段，交给回调。"""
        self._cancel_turn_check()
        if not self._buf:
            self._reset_run()
            return

        buf = self._buf
        speech_samples = self._speech_samples

        # 先重置状态 —— 回调可能同步触发新一轮累积
        self._buf = []
        self._reset_run()

        # 把上一段退回来的音频接在前面。见下面短段分支的解释。
        # 注意顺序：**先判这一段自己够不够格，再接 carry**。反过来的话
        # 几段噪声攒着攒着就能凑够 min_seg，自己把自己送进 ASR——展台上
        # 表现为没人说话时屏幕上蹦出一句乱码。carry 只能搭车，不能开车。
        speech_s = speech_samples / SR
        if reason == "endpoint" and speech_s < self._min_seg_s:
            # 太短的**顺延到下一段**，不丢。判据是实际说话时长，不是缓冲总长
            # ——缓冲里含着触发提交的那段尾部静音，用总长会让「咳一声 +
            # 0.6s 静音」蒙混过关。
            #
            # 以前这里是直接丢。丢掉的理由是过滤咳嗽和语气词，听起来合理，
            # 但实测代价大得多：展台素材 03 的第一句「各位专家，」后面 TTS
            # 停了 0.58s，超过 min_silence（0.45s），于是它自成一段；而它
            # 本身只有 0.45s 语音，不足 min_seg（0.60s），就被**静静删掉**
            # 了。屏幕上那句话从「各位专家，下面介绍…」变成「下面介绍…」
            # ——观众看不出丢了字，讲解员也不知道，因为没有任何一处报错。
            # 真人说话时「各位好，」「那么，」这种开场白正是这个形状，
            # PT 设备上同样会中招。
            #
            # 顺延的做法：暂存到 _carry，下一次提交时接在前面，两段合成一句
            # 送 ASR。**不能塞回 _buf** ——_buf 是静音状态机的工作区，塞回去
            # 那些静音帧会让下一轮的 feed() 以为「已经开口」，_in_speech 提前
            # 置位，静音阈值就再也触发不了（这个坑踩过）。所以必须是独立的
            # 一块。
            log.debug("segment too short (speech=%.2fs) — 顺延到下一段", speech_s)
            self._carry.extend(buf)
            self._trim_carry()
            return

        if self._carry:
            buf = self._carry + buf
            self._carry = []
            self._carry_speech = 0

        audio = np.concatenate(buf)

        # 裁掉多余的尾部静音，但保留一点给 ASR 收尾
        keep = self._keep_trailing * FRAME_SAMPLES
        if len(audio) > keep * 2:
            audio = audio[:len(audio) - keep]

        seg = CommittedSegment(audio, reason)
        log.info("committed %.2fs segment (%s, speech=%.2fs)",
                 seg.duration_s, reason, speech_samples / SR)
        try:
            self._on_segment(seg)
        except Exception as e:
            log.error("on_segment callback failed: %s", e)

    def _trim_carry(self) -> None:
        """把顺延缓冲裁到上限以内，并重新数一遍里面有多少语音。

        裁头不裁尾：顺延的这段要接在**下一句前面**，离下一句越近的越可能是
        真正的开场白，越早的越可能是上一轮留下的噪声。

        语音样本数是**重新数**的，不是减出来的。只在这里用了一次逐帧 RMS，
        但省不得：flush 靠这个数判断「顺延里还压着话吗」，而裁掉的帧里含
        多少语音是未知的——按比例估会越估越偏，最后表现为松手时冲出来一段
        谁也没说过的噪声。carry 上限 2 秒、一共 63 帧，数一遍不值一提。
        """
        total = sum(len(f) for f in self._carry)
        while len(self._carry) > 1 and total > self._max_carry_samples:
            total -= len(self._carry.pop(0))
        self._carry_speech = sum(
            len(f) for f in self._carry
            if len(f) and float(np.sqrt(np.mean(f ** 2))) > self._sil_thresh)

    def _reset_run(self) -> None:
        self._sil_run = 0
        self._in_speech = False
        self._speech_samples = 0
        self._speech_ring = []

    # ── 收尾 ────────────────────────────────────────────────────────────────

    async def flush(self) -> None:
        """ptt_end 或会话结束：把残留缓冲冲出来，一个字不丢。"""
        if self._pending_turn_check and not self._pending_turn_check.done():
            try:
                await asyncio.wait_for(self._pending_turn_check, timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
        if self._buf:
            # _commit 会自己把 _carry 接在前面，这里不用管
            dur = (self._total_samples()
                   + sum(len(f) for f in self._carry)) / SR
            if dur >= self._min_seg_s:
                self._commit("flush")
            else:
                # 短尾段：如果有实际语音内容也提交（用户就说了两个字）
                if (self._speech_samples + self._carry_speech) / SR >= 0.25:
                    self._commit("flush")
                else:
                    self._buf = []
                    self._carry = []
                    self._carry_speech = 0
                    self._reset_run()
        elif self._carry_speech / SR >= 0.25:
            # 缓冲空了但顺延里还压着话——最后一句就是那句「太短」的。
            # 不在这儿冲出来，它就永远等不到下一班车了。
            self._buf = self._carry
            self._carry = []
            self._speech_samples = self._carry_speech
            self._carry_speech = 0
            self._commit("flush")
        else:
            self._carry = []
            self._carry_speech = 0

    def reset(self) -> None:
        self._cancel_turn_check()
        self._buf = []
        self._carry = []
        self._carry_speech = 0
        self._speech_ring = []
        self._reset_run()

    @property
    def buffered_s(self) -> float:
        return self._total_samples() / SR

    @property
    def sil_thresh(self) -> float:
        """静音判据。报出去是给「什么都没听到」那条诊断用的——

        现场只知道「按了没反应」，而真正的原因可能是话筒电平低于这个数，
        一帧都没过线。把阈值和实测电平一起报给客户端，日志里一眼能看出
        是「没说话」还是「说了但太轻」。这两件事的处置完全不同。
        """
        return self._sil_thresh

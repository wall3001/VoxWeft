# DEVLOG — what broke, what we were wrong about, and what it cost

> This file is a first-class deliverable, not an appendix. A working demo can
> be written by any LLM; *where it went wrong and why* cannot. Every entry
> below was hit for real on M2 / M3 Ultra hardware between 2026-08 and 2026-09.
> Each entry: **symptom → root cause → fix → what it cost us**. Entries marked
> ⚠️ are the ones most likely to bite you again.

Structure per entry follows the triage we ended up using for everything:
**confirmed / disproved / unverified**. Writing down what we *disproved*
turned out to matter as much as what we fixed — it is the only thing that
stops the next person from re-running the same dead-end investigation.

---

## 1. MLX

### ⚠️ 1.1 GPU stream is thread-bound — load and infer on the same thread

**Symptom.** `There is no Stream(gpu, 0) in current thread`, raised from the
first inference call. Loading succeeded; inference failed.

**Root cause.** MLX's default GPU stream belongs to the thread that created
it. A model loaded in thread A (e.g. a FastAPI startup hook or a warm-up
thread) and run in thread B (a request handler's `run_in_executor` default
pool) is on the wrong stream.

**Fix.** One `ThreadPoolExecutor(max_workers=1)` **per model**. Both the load
*and* every inference are submitted to that executor, so they are trivially on
the same thread. TTS and ASR each get their own executor so a long TTS
generation does not queue ASR behind it. See `voxweft/runtime/bridge.py`.

**Cost.** Half a day, because the error message points at the inference site
while the mistake is at the load site.

### ⚠️ 1.2 Holding a lock across an async-generator `yield` deadlocks the whole process

**Symptom.** Bridge process alive, CPU ~50%, `/health` returns ok, `/ready`
times out, zero log lines. Both TTS and ASR dead simultaneously.

**Root cause.** The first serialization attempt was a single `_infer_lock`
around inference. The streaming TTS handler is an async generator: it
acquires the lock, `yield`s a chunk, and is suspended *still holding the
lock* while the client reads. An ASR request then waits on the same lock
forever, and the event loop has nothing to run.

**Fix.** Remove the lock entirely. The single-thread executors from 1.1
already give mutual exclusion, and `await executor.submit(...)` releases
control between chunks instead of pinning it.

**Cost.** This one was misdiagnosed for a day as "a particular Chinese
audio clip is poisonous" because the deadlock happened to be triggered by
a request that arrived while a stream was open.

### 1.3 Two threads constructing the same MLX model ⇒ segfault, process gone

**Symptom.** Bridge disappears with `exit -9`. launchd restarts it; it
disappears again on the next warm request.

**Root cause.** Warm-up ran in a background thread so that uvicorn could
start immediately. uvicorn started accepting requests immediately. If a real
request arrived during warm-up, both paths entered the lazy-load code and
constructed two model instances concurrently. MLX is not thread-safe here.
The original guard was an `asyncio.Lock`, which is per-event-loop and does
nothing across threads.

**Fix.** `threading.Lock` around load, shared by warm-up and request paths.
Warm-up goes through the *same* `_ensure_*_locked()` the requests use — no
private construction path.

### 1.4 `mlx_lm` 0.31 changed the sampling API

**Symptom.** `TypeError: generate() got an unexpected keyword argument 'temp'`.

**Root cause.** `mlx_lm.generate(..., temp=x)` was removed in 0.31.x in
favour of `sampler=make_sampler(temp=x)`.

**Fix.** `from mlx_lm.sample_utils import make_sampler` and pass `sampler=`.
Pin `mlx-lm>=0.31` in `pyproject.toml` and do not support both.

### 1.5 `mx.eval` placement decides whether streaming is actually streaming

Latents kept on-device (`mx.transpose` instead of `np.array(...).T`) and a
single device→host copy at the end saved a measurable slice of RTF in the
non-streaming path. In the streaming path the opposite holds: you *must*
`mx.eval` each chunk before yielding it or the first audio is delayed until
the graph is forced later. This is why `generate()` and `generate_streaming()`
in `voxweft/core/generate.py` deliberately look different.

---

## 2. VoxCPM2 model behaviour

### ⚠️ 2.1 `temperature` is not a sampling temperature

**Symptom.** Trying to reduce output variance by lowering `temperature`.
At `t=0.01` generation never emits `stop`, runs to the 80 s cap, and 80% of
frames are silence.

**Root cause.** In VoxCPM2 `temperature` scales the **initial noise of the
flow-matching ODE** (`z = randn(...) * temperature`), not a softmax. Near
zero the CFM has no stochasticity to work with and does not converge.

**Fix.** Leave it at the official `1.0`. Documented in `TTS-FINDINGS.md` §2.
This knob *looks* like it treats jitter; it only makes things worse.

### 2.2 Voice design captions: measured leak, then measured no-leak

VoxCPM2's documented voice-design mode puts a natural-language description
in parentheses at the start of the text — `(A young woman, gentle voice)Hello`
— and the description is not spoken.

**What we saw on the stage build (2026-09-12).** With `(一位沉稳清晰的播音员)`
prepended by the runtime, every utterance began with the description spoken
aloud. ASR round-trip: 4.80 s with prefix and the words in the transcript,
2.56 s without. We removed the prefix and wrote "the model reads everything".

**What we measure now** (`scripts/design_voices.py` and a direct probe, same
weights, current port, 2 runs each): no leak. Chinese caption, English
caption, with a cloned reference or zero-shot — the transcript is the body
only, and the voice follows the description well enough to build a showcase
from it (§ Appendix, `samples/voices`).

**What changed in between.** The port was rewritten (streaming decoder,
device-side latents, tokenizer path) and the ASR used for the round-trip was
a different model. We could not reproduce the leak on the old build because
the box is at the venue. Root cause of the September leak: **not
established**. What is established: on this port the mechanism works, and
the runtime still does not add captions — voices are *designed* offline with
a caption, the chosen sample becomes a clone reference, and synthesis text is
never altered. That keeps the "TTS is a dumb renderer" contract regardless
of which way the model behaves on a given build.

### ⚠️ 2.3 The "first-frame pop" — what it was and what it wasn't

**Symptom.** Audible click at the start of streamed audio, reported on the
M3 stage box.

**What we believed.** The first one or two patches carry a VAE/CFM onset
transient; skip emitting them (`warmup_patches=2`, following mlx-audio's
VoxCPM2 implementation). This shipped in the stage build.

**What we measured** (`scripts/probe_onset.py`, M2, 6 steps, 4-bit, zero-shot
and clone, 3 runs each, `warmup=0 fade=0`):

| | patch 1 RMS | patch 2 RMS | rest RMS | max adjacent-sample jump in patches 1–2 | signal peak |
|---|---|---|---|---|---|
| zero-shot ×3 | 0.004 / 0.112 / 0.061 | 0.015 / 0.085 / 0.073 | 0.070 / 0.067 / 0.036 | 0.024 / 0.057 / 0.073 | 0.5–0.8 |
| clone ×3 | 0.002 / 0.001 / 0.052 | 0.122 / 0.023 / 0.107 | 0.068 / 0.049 / 0.054 | 0.051 / 0.028 / 0.101 | 0.4–0.6 |

No transient. Patch 1 is usually *quieter* than the body (speech starts from
natural silence); the largest sample-to-sample jump in the first two patches
is 0.02–0.10 against peaks of 0.4–0.8 — normal speech slope, not a click. The
first millisecond peaks at 0.0014: the streaming VAE starts from zero cleanly.

**What `warmup_patches=2` actually did.** Two things, both bad on M2:
1. Started playback mid-speech, so the first emitted sample is at speech
   amplitude — first-1 ms peak **0.0437** — i.e. it *creates* the step it was
   meant to remove.
2. Cost **~470 ms** of first-audio latency (warm 304 ms → 773 ms), not the
   160 ms of audio it skips, because on a base M2 each patch is ~230 ms of
   wall time. On M3 Ultra (RTF 0.2) the cost is small, which is why nobody
   noticed on the stage box.

**Fix.** `warmup_patches=0` (default), `fade_in_ms=8` (default): an 8 ms
raised-cosine on the first emitted chunk. With the fade, first-1 ms peak is
0.0002 whatever the stream start. The knob stays for anyone who wants to
reproduce mlx-audio's behaviour.

**Where the stage click most likely came from.** The client-side jitter
buffer (§3.3): fragments scheduled at WebSocket message boundaries, not at
200 ms chunk boundaries. That fix was written but not verified on the box
before it shipped to the venue; the server-side fade is belt-and-braces.

**Lesson.** We inherited a default from a reference implementation and
documented a mechanism for it that we had not measured. The measurement
took 20 minutes.

### 2.4 4-bit quantization is not the source of variance

Disproved. `quantize_lm=4` vs bf16, 5 runs each: duration jitter 1.44 s vs
1.76 s, SNR 29.0 dB vs 26.6 dB. bf16 is not better. The variance that *is*
there (same text, same params, 3.84–6.40 s duration across 6 runs) is the
model sampling normally without a seed. Keep the 4-bit; keep the memory.

### 2.5 Acronyms are read as words

TTS round-trip on the intro text: "We do not compete on BLEU" came back as
"We do not compete on being young" — VoxCPM2 read `BLEU` as the French word,
the ASR heard what it heard. Text normalization (spell out or hyphenate
acronyms before synthesis) is the client's job; the runtime does not do it.
CER on that sentence: 40%. Everything else in the set: 0–9%.

---

## 3. Streaming TTS engineering

### 3.1 Cold start: 2149 ms first chunk vs 62 ms warm

**Root cause.** Lazy singletons — first request paid weight load + graph
compile.

**Fix.** Warm-up on process start that runs the *real* inference path once
(not just weight load: compilation and caches are built on first inference).
Runs in a background thread so the port is up for health checks within
seconds. First playable chunk after warm-up: 124–168 ms.

**Measurement discipline.** Two latency figures were being confused:
HTTP first byte (~50 ms earlier) vs first *playable* chunk on the client.
Only the second one is the north-star number. Say which one you mean.

### 3.2 Streaming VAE: stateful causal convs, bit-identical to full decode

The streaming decoder carries causal-conv state between chunks. Verified:
same latents through per-frame streaming, all-at-once streaming, and
non-streaming full decode produce identical output (0.0% difference).
This is what makes 0.51 s first-audio possible vs 9.3 s for the upstream
non-streaming path — and it is *not* where any audible artefact comes from.
Golden test: `tests/test_core_golden.py::test_streaming_equals_full`.

### ⚠️ 3.4 Loading a second model evicts the first one's buffer cache

**Symptom.** Bridge first real TTS request 720–762 ms first byte; every
request after it 305 ms; in-process streaming on the same machine 286 ms.
Warm-up had run and `/ready` was 200.

**Disproved first.** "Warm-up didn't exercise the full path" — we changed
warm-up from 3 patches of "你好" to a full 23-patch sentence. First request
still 720 ms. Not compile.

**Root cause.** Order of warm-up: TTS loaded and primed, *then* ASR (another
~3 GB) loaded. On a 16 GB M2 the second load evicts the first model's MLX
buffer cache; the next TTS inference re-allocates. A plain ASR *inference*
afterwards does not evict again (measured: 301 ms TTS after an ASR call) —
it is specifically the load.

**Fix.** After all models are loaded, run one short TTS generation
(`_reprime_tts`, ~2.3 s at start-up, once). First request: 311 ms. Any
process that hosts more than one MLX model should prime *after the last
load*, not after each one.

**Cost.** 450 ms on every cold session for as long as the stage build ran,
invisible on M3 Ultra because 512 GB never evicts anything.

### 3.3 Client jitter buffer: periodic "click" was a scheduling artefact

Server sends 200 ms chunks; WebSocket message boundaries do not align with
them. Scheduling each received fragment as its own `AVAudioPCMBuffer` makes
the player node stitch at every fragment boundary — audible as periodic
clicks. Fix: accumulate to 200 ms before `scheduleBuffer`. Also:
`player.stop()` must be immediately followed by `play()` or later buffers
queue without playing.

---

## 4. Endpointing / segmentation

### ⚠️ 4.1 SmartTurn v3 is net-negative for simultaneous interpretation

**Expectation.** A learned end-of-turn model would beat RMS silence.

**Measurement.** Same Lao conference audio: silence rule → 28 segments;
SmartTurn on → 8 segments, all exactly 15 s = every one of them was the
forced cut. The model answered "not finished" on almost every check, blocked
the silence cut, and everything fell through to `max_utterance`.

**Why.** SmartTurn is a *dialogue* model: "has the other person stopped and
is waiting for me?" In continuous speech the speaker is *not* waiting, so it
is correctly saying "not done" — and that is the wrong question for
interpretation, where we want to cut at clause boundaries while they talk.

**Decision.** Off by default (`VOXWEFT_SMART_TURN=0`). Kept as an option for
the dialogue mode. Before enabling anywhere, compare segment counts on the
same audio.

### 4.2 Three SmartTurn integration traps (if you do turn it on)

1. `append_audio()`'s return value is *not* the model verdict — it is a
   silence-duration fallback. The verdict is `analyze_end_of_turn()`.
2. Therefore never feed it silence frames: 118/118 COMPLETE.
3. Feeding the whole accumulated buffer (15 s, several sentences) ⇒ 26/26
   INCOMPLETE. Feeding the buffer *tail* ⇒ it is all silence by the time
   you trigger, 96 ms of nothing. You need a separate rolling ring of the
   last ~4 s of *speech* frames. That is `_speech_ring` in the segmenter.
4. One verdict is ~50 ms, not ~10 ms. Thread pool, and only once silence
   is already established.

### 4.3 Fixed RMS threshold drops words when the microphone changes

`sil_thresh=0.035` was tuned for a close-talk stage mic. On phone
recordings the median speech-frame RMS is 0.02 — 70% of speech frames were
classified as silence. Commercial ASR CER on the same material: 4.7% → 16.3%.

**Fix.** Adaptive threshold (`adaptive=True`): track a noise floor and
clamp the effective threshold to `[floor, sil_thresh]`; soft forced cuts
look for the quietest frame in a window instead of cutting at an arbitrary
sample. CER back to 10% on phone audio; 12/12 segmenter tests pass.

### 4.4 Segments too short were silently deleted — including real words

"各位专家，" (0.45 s of speech) followed by a 0.58 s pause became its own
segment, failed `min_seg_s=0.60`, and was **discarded**. The screen showed
"下面介绍…" instead of "各位专家，下面介绍…". No error anywhere. Real
speech has exactly this shape ("So,", "Okay,", vocatives).

**Fix.** Short segments are *carried* into the next segment (`_carry`), not
dropped. They must live in a separate buffer — putting them back into `_buf`
makes the silence state machine think speech already started and the
threshold never triggers again. Carry is capped at 2 s and its speech
content is re-counted, not estimated.

### 4.5 `asyncio.ensure_future` silently binds to the wrong loop

**Symptom.** SmartTurn never called once; only forced cuts.

**Root cause.** `ensure_future` takes "the current event loop", which is the
right one when called from inside a loop callback and a *wrong/closed one*
when called from a thread or early in startup. The task is created, never
runs, never errors.

**Fix.** `asyncio.get_running_loop()` explicitly; on `RuntimeError`, fall
back to the silence rule and log it.

### 4.6 Do not `Task.cancel()` a not-yet-started coroutine to invalidate it

Cancelling a coroutine that has not begun executing shows as `done()` but
never ran; the scheduler then sees "no pending check" and schedules another,
racing the cancel. Use a generation counter (`_turn_check_gen`) and let the
task discard its own result.

### 4.7 The config-ordering bug that produced garbage

**Symptom.** Lao audio recognized as Chinese: "拜拜你他你他".

**Root cause.** Client pushed `session_config` mid-utterance; the
orchestrator was rebuilt with the wrong source language for audio already
in flight.

**Fix.** Config changes rebuild the orchestrator only between presses; the
Swift client gates on it. Any client you write must not send
`session_config` between `ptt_start` and `ptt_end`.

---

## 5. Translation

### 5.1 "Repetition means the model is too small" — wrong

**Claim.** NLLB-1.3B produces repetitive output on short fragments; 3.3B
would fix it.

**First test (7 hand-written samples).** 1 repetition. Concluded "decoding
params, not size". *Premature — sample too small.*

**Second test (28 real ASR fragments from production audio).** 0/28
repetitions on **both** 1.3B and 3.3B. The repetition people had seen was
the routing bug in 4.7 feeding wrong-language text into the translator.

**Decision.** Keep 1.3B where NLLB is used at all. Record this as a
reminder that a 7-sample test can only tell you a hypothesis is *not
obviously false*.

### 5.2 Small LLMs guess the wrong language for zh→low-resource

A 4B chat model asked for Lao would sometimes emit Thai, or Lao script
with Thai vocabulary, confidently. NLLB never does this. Hence the router:
zh→SEA via NLLB or a dedicated MT model; SEA→zh via the LLM (where 4B is
strong: 267–310 ms per sentence). See `voxweft/runtime/router.py`.

### 5.3 "thinking" mode wrecks translation quality

With a reasoning-enabled chat model, letting it think before translating
made translations *worse* — it rationalises, then "improves" the sentence.
Disable thinking for MT, or train a small LoRA that suppresses it.

### 5.4 Glossary injection: pre-model pinning vs post-model patching

Proper nouns go in *before* the model as target-script substitutions so NLLB
copies them; common words pinned this way get copied in Chinese word order
and produce broken sentences. Post-model patching is only for terms the
model copied back untranslated. Keep both tables short — a long glossary in
the prompt biases short-sentence decoding.

---

## 6. ASR

### 6.1 Only send a language label the loaded model lists

mlx-audio passes an unknown `language=` name straight into Qwen3-ASR's
decoder prefix (`language Lao<asr_text>`). A label the model was never
trained on yields plausible output in the wrong script, silently. The
bridge checks `config.support_languages` and falls back to auto-LID for
anything not listed; the router sends Lao/Khmer/Burmese to Whisper-large-v3,
which has them, instead of to stock Qwen3-ASR, which does not.

### 6.2 Auto language-ID on short clips is unreliable between Lao and Thai

Do not rely on LID for segments of 1–3 s. Always pass an explicit language
from the session.

### 6.3 Hotwords: a few high-value proper nouns beat a long list

"AI" in Lao was heard as the single letter "ເອ", and the translator then
"repaired" the whole clause into nonsense. Adding the two Lao forms of "AI"
fixed it at the source. But long hotword lists perturb decoding on short
segments; keep them sparse. Chinese homophone traps that were real:
端侧→端测, 切段→切断, 出声→出生.

### 6.4 The ASR bridge decoded WAV headers as audio — for the whole stage run

**Found by** code review after open-sourcing, not by listening.

The runtime sends each segment as a base64 RIFF WAV; the bridge did
`np.frombuffer(wav_bytes, dtype=np.int16)` on the entire payload. The 44-byte
header became 22 samples at up to ±0.98 — a 1.4 ms full-scale click glued to
the front of every ASR input, for every language, throughout the exhibition.
Recognition still worked (the models are robust to 1 ms of noise), which is
exactly why nobody noticed. Fix: decode with `soundfile`, resample if needed.

**Also caught in the same review** and fixed in commit 3: the shipped default
ASR (stock Qwen3-ASR-1.7B) was being forced to decode Lao/Khmer/Burmese
under labels it never saw — the router still described the private
fine-tune; the NLLB fallback returning `{"text": "", "error": …}` was
emitted as an empty translation with no `error` and no `turn_end`; a TTS
exception mid-stream ended as a clean HTTP 200; `text_input` in agent mode
never synthesized; the adaptive threshold the README advertised was never
enabled in the runtime; the Swift client wrote every ASR result into the
*last* turn instead of matching `seq`. Six of ten were "the docs describe
the fix, the code doesn't have it" — which is what happens when the devlog
is written before the extraction is finished.

---

## 7. Build & deploy on macOS

### ⚠️ 7.1 SwiftPM incremental build reports success without compiling

**Symptom.** `swift build` → "Build complete (2.8s)". App runs old code.
Days of "I can see the log line in the source, the user sees nothing".

**Root cause.** Files copied via `scp` keep an *older* mtime than the
existing build artefacts, so SwiftPM decides nothing changed.

**Fix.** `rm -rf .build` before every deploy. A real compile of this app is
40+ s; if you see 2.8 s, nothing was built. Do not use `strings | grep` on
the binary to check whether Chinese literals made it in — it misses UTF-8;
use Python `b"...".encode("utf-8") in data`.

### ⚠️ 7.2 macOS 26 (Tahoe) + Swift 6.3: SwiftUI `Button` aborts on click

`_ButtonGesture.internalBody → MainActor.assumeIsolated →
swift_task_isCurrentExecutorWithFlags → _objc_fatal → SIGABRT`. Every
`Button` in the app. Workaround: replace with a `DragGesture`-based
`.tappable` modifier or `onTapGesture`. (First attributed to
`SystemSegmentedControl`; that was wrong — it was `Button` all along.)

### 7.3 `Task { @MainActor in }` created inside `SwiftUI.Update.dispatchActions()` never runs

Not cancelled, just never executed. `DispatchSourceTimer` also fails there.
`Timer.scheduledTimer` on `RunLoop.main` in `.common` mode works.

### 7.4 Deleting a "leftover" `stopFlag = false` broke the demo player

It was the reset. The task exited immediately and the generation counter
advanced by 2–3 per press instead of 1. If a flag reset looks redundant,
find who reads it before deleting it.

### 7.5 Command Line Tools ≠ a Swift toolchain

On one M2 with CLT only (`swiftlang-6.0.3.1.10`), the SDK was built by
`swiftlang-6.0.3.1.5`: SwiftPM could not even link `Package.swift`
(undefined `PackageDescription.Package.__allocating_init`), and plain
`swiftc` failed on `CoreFoundation` with "this SDK is not supported by the
compiler". Nothing in the project is at fault; the CLT install is internally
inconsistent. `app/build.sh` exists so a `swiftc`-only path is available, but
if you see those two errors, reinstall CLT or use a machine with Xcode.

### 7.6 `setVoiceProcessingEnabled(true)` can make the whole engine fail to start

On a MacBook Pro whose built-in mic array reports 3 input channels, enabling
voice processing reshapes the input to 7 channels and `AVAudioEngine.start()`
fails with `-10875` at `kAUInitialize` on the *output* unit — so neither
capture nor playback works, and the only visible symptom is an empty level
meter and silence. Works fine on the M3 Ultra's 1-channel input. The app now
tries with VP, and on failure rebuilds the engine without it (no AEC, so
headphones). Reproduced with a 15-line standalone probe; the first
diagnosis ("TCC denied the mic") was wrong — TCC had granted it.

### 7.7 Health probes: `/health` ≠ `/ready`

`/health` is "process alive" and is true during the 1.7 s cold start;
`/ready` is "models loaded and warmed". Watchdogs and client start-up
checks must probe `/ready`, or users eat the cold start.

### 7.8 An interrupt that reached nothing

Stop button "did nothing": audio kept playing, and the next sentence waited
~20 s. Four layers each dropped the ball:

1. `interrupt()` set a cancel `Event` and immediately replaced it with a
   fresh one; the consumer, parked on the next chunk, only ever saw the new
   unset Event. → generation counter compared per chunk.
2. Text turns ran as untracked `ensure_future` tasks outside the worker. →
   tracked in `_side_tasks`, cancelled on interrupt.
3. Cancellation raised `CancelledError` at `await aq.get()`, skipping the
   cut branch entirely, so the producer thread and its HTTP stream lived on.
   → `except CancelledError: abort()` around the consumer.
4. `requests.Response.close()` on a half-read stream returns the connection
   to the pool; the socket stays open and the bridge never learns. And the
   bridge's own generator sat in `await aq.get()` between 300 ms patches, so
   even a real close was noticed one patch late while the single TTS thread
   finished the whole sentence. → shut the raw socket; bridge polls
   `request.is_disconnected()` at 100 ms and sets a stop flag the producer
   checks per patch.

Measured: next sentence after interrupt 19.9 s → 1.2 s on M2.

### 7.9 Playback stutter that was not a sample-rate problem

Fine-grained choppiness on M2 playback ("like the sample rate is wrong"):
formats matched (48 kHz mono float both sides). The server produces at RTF
1.47, i.e. a 200 ms chunk every ~300 ms; the player drained each chunk and
sat silent ~100 ms until the next — a gap five times a second. Fix in the
client: 0.6 s pre-roll before starting, and drop back into buffering
whenever the scheduled queue runs dry, so a slow producer costs one pause
per utterance instead of a stutter per chunk. On real-time hardware the
pre-roll is a one-time 0.6 s.

### 7.10 Non-daemon executor threads make restarts slow

`ThreadPoolExecutor` threads are non-daemon; without an explicit
`shutdown(wait=False, cancel_futures=True)` on process exit, launchd
`kickstart` waits several seconds.

---

### 7.9 The 6–14 s that were not ours

First E2E on the M2 dev box: 6–14 s from segment commit to first audio.
Bare models, same box, minutes later: ASR 0.95 s + LLM 0.87 s + TTS 0.3 s
≈ 2.1 s. The difference was the machine — Lark (1.2 GB), a VPN client
(0.8 GB), two leftover Python processes, and our own previous run still
draining its serial queue and transcribing the silence after the clip
("Oh." "Shh." — ASR hallucination on tail noise, not a bug in the clip).
Free memory was 100 MB. After `pkill`: 9.8 GB free, 2.5 s commit-to-audio.
Rule: `vm_stat` before every latency claim; kill the previous run before the
next one.

## 8. Measurement mistakes we made (so you don't)

* Concluding from 7 samples (§5.1).
* Trusting a code comment over an ASR round-trip (§2.2).
* Confusing HTTP-first-byte with first-playable-chunk (§3.1).
* Checking deployment with `strings | grep` on UTF-8 (§7.1).
* Attributing a crash to the most visible frame in the stack (§7.2).
* Attributing a deadlock to the input that happened to trigger it (§1.2).
* Fixing 'Stop does nothing' at one layer when four were broken (§7.8).
* Reporting E2E latency from a machine with 100 MB free and a previous run still draining (§7.9).

---

## Appendix — numbers that anchor this repo

| what | M2 16 GB | M3 Ultra |
|---|---|---|
| VoxCPM2 4-bit, 6 CFM steps, RTF | 1.48 (streaming) / 1.35 | ≈ 0.2 |
| first audio, streaming, warm (`voxweft say --stream`) | 304 ms zero-shot · 403 ms clone | 124–168 ms |
| first audio, cold (includes `mx.compile`) | ~700 ms | 2149 ms → 124 ms with warm-up |
| upstream non-streaming first audio | 9.3 s | — |
| cold start first chunk (no warm-up) | — | 2149 ms |
| TTS ASR-round-trip intelligibility (zh/vi/lo/km) | ✅ verbatim | ✅ |
| segmenter, phone audio, fixed vs adaptive threshold (ASR CER) | 16.3% → 10% | — |

RTF 1.48 on a base M2 means the audio is produced at 2/3 of playback speed:
streaming will stutter on that machine. That is the hardware, not the port —
the upstream implementation measures the same RTF. On M3 Ultra it is
real-time with margin. See `docs/HARDWARE.md`.

---

## Open — measured but not yet explained

* **Open ASR for Lao / Khmer / Burmese is not there yet.** Stock Qwen3-ASR
  does not list them; Whisper-large-v3-turbo has them on paper but on our
  clips returns Thai script for Lao and repetition loops for Khmer (bench:
  lo CER 89%). Meta's Omnilingual-ASR CTC-1B (Apache-2.0, sherpa-onnx,
  CPU) measured lo CER ~11% and is the honest open choice; an adapter for it
  is the next runtime backend to write. Until then the router sends these
  languages to Whisper because it is the only thing that will not silently
  decode them under a wrong label.
* **Loudness variance between runs** (same text/params: whole-signal RMS
  0.006–0.11, peaks up to 0.97). Known model behaviour without a seed
  (`TTS-FINDINGS.md` §3); a peak of 0.97 is one bad run from clipping.
  A light limiter on the output path would be cheap; not done.
* **M-series Pro/Max RTF** is unmeasured (`docs/HARDWARE.md`). We only have a
  base M2 and an M3 Ultra.

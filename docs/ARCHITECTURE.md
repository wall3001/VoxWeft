# Architecture

## Two processes, and why

```
 mic ──int16 16k──▶ ┌──────────────── voxweft serve (:3100) ───────────────┐
                    │ IncrementalSegmenter ─▶ Orchestrator ─▶ router        │
                    │        │                    │  ├─ ASR  ──HTTP──▶ ┐     │
                    │   RMS / adaptive             │  ├─ MT/LLM (mlx_lm, in-process)
                    │   0.45 s silence             │  └─ TTS  ──HTTP──▶ ┤     │
                    └─────────────────────────────────────────────────┼─────┘
                                                                      ▼
                    ┌──────────────── voxweft bridge (:8133) ─────────────────┐
                    │  _tts_executor (1 thread) ── VoxCPM2Pipeline (MLX)       │
                    │  _asr_executor (1 thread) ── Qwen3-ASR (mlx_audio)       │
                    │  /generate stream=true → int16 48 kHz chunks             │
                    │  /v1/chat/completions   → <asr_text>…                    │
                    │  /health (alive)  /ready (loaded + warmed)               │
                    └──────────────────────────────────────────────────────────┘
```

MLX GPU streams are bound to the thread that created them and the models are
not thread-safe. The bridge gives each model exactly one thread for both load
and inference. The server process never touches those models; its only MLX
work is the in-process LLM/MT, which has its own single call site.

Merging the two processes is possible but every attempt at "one process, a
lock" produced either the stream error or a deadlock (DEVLOG §1.1, §1.2).

## Segment lifecycle (interpreter mode)

```
feed(frame 32 ms) ──▶ speech? ──yes──▶ accumulate, reset silence run
                        │ no
                        ▼
                 silence ≥ 0.45 s ? ──no──▶ (long utterance ≥ 6 s → soft forced cut)
                        │ yes
                        ▼
              [SmartTurn on?] ──▶ async verdict ──COMPLETE──▶ commit
                        │ off                      └─INCOMPLETE─▶ recheck in 0.4 s
                        ▼
                     commit ──▶ speech < 0.6 s ? ──yes──▶ carry into next segment
                                       │ no
                                       ▼
                             CommittedSegment ──▶ worker queue (serial)
                                       │
                          ASR ──▶ glossary/canon ──▶ MT ──▶ TTS stream ──▶ client
```

The worker is serial on purpose: one synthesis at a time keeps the client's
single playback node coherent, and a `ptt_start` interrupt only has one
in-flight turn to cancel.

## Language router

| direction | backend | why |
|---|---|---|
| SEA → zh | LLM | 4B is strong here; 267–310 ms/sentence |
| zh → SEA | dedicated MT (NLLB / Hy-MT2 / MiLMMT) | small LLMs guess the wrong script |
| SEA ↔ SEA | dedicated MT, direct | two hops double latency and compound errors |
| anything else | LLM | |

ASR backend by language: SEA-finetuned Qwen3-ASR for {lo, th, km, my, vi, zh}
(pass the **lowercase training label**), Whisper-MLX for the rest.

## Modes

* `interpreter` — transcript → translation → speech. No persona.
* `agent` — transcript → LLM with history → sentence-streamed speech. Same
  segmenter, same events; `llm_sentence` replaces `translation`.

## Events

See the docstring at the top of `voxweft/runtime/server.py`. Binary frames
in are int16 mono 16 kHz with arbitrary framing; binary frames out are
int16 mono 48 kHz in 200 ms chunks between `audio_start` / `audio_end`.

# Hardware tiers

Measured, not projected, unless marked.

| tier | machines | TTS (VoxCPM2 4-bit, 6 steps) | experience |
|---|---|---|---|
| **recommended** | M3 Ultra · M4 Max · M2 Ultra | RTF ≈ 0.2, first audio 124–168 ms warm | full pipeline real-time, ASR→MT→first TTS chunk ≈ 0.85 s |
| usable | M-series Pro/Max (non-Ultra) | est. RTF 0.6–1.0 (**unmeasured**) | ASR/MT real-time; TTS near real-time |
| runs | M2 / M3 base, 16 GB | RTF 1.48 streaming, first audio ~300 ms | ASR/MT real-time; TTS produces audio slower than playback — streaming stutters. Full text→speech turn with Qwen3-4B translation: 2.35 s median to first audio (the 4B model, not the TTS) |

On a base M2 both our implementation and upstream mlx-audio's measure the
same RTF 1.5: the ceiling is the chip, not the port. What the port buys on
that machine is first audio 18× sooner (0.5 s vs 9.3 s).

## Memory (16 GB base M2, all resident)

Qwen3-ASR-1.7B 8-bit 2.2 GB · Qwen3-4B 4-bit 2.8 GB · VoxCPM2 4-bit 3.0 GB
≈ 8 GB. Add Hy-MT2-1.8B 4-bit (1.3 GB) or NLLB-1.3B for MT if you route
zh→SEA. Lazy loading per language pair is not implemented; everything you
configure is loaded at warm-up.

## Not supported

* Intel Macs, Linux, Windows — MLX only.
* Ollama / llama.cpp as the LLM backend — latency budget does not close.

## Measuring on a shared machine

Numbers taken while other software is resident are not numbers. On the 16 GB
M2 the pipeline needs ~8.5 GB for its three models; with a chat client and
a VPN helper holding another 2 GB the unified memory thrashed, free fell to
~100 MB, and every stage was 3–4× slower with nothing in the logs to say
why. Kill everything you can (`pkill` the chat apps, check `vm_stat` shows
several GB free) before trusting any latency figure, and check that no
earlier run is still draining its queue — a serial worker will happily keep
transcribing whatever is left over, which looks like hallucination.

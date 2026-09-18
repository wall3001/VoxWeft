# Hardware tiers

Measured, not projected, unless marked.

| tier | machines | TTS (VoxCPM2 4-bit, 6 steps) | experience |
|---|---|---|---|
| **recommended** | M3 Ultra · M4 Max · M2 Ultra | RTF ≈ 0.2, first audio 124–168 ms warm | full pipeline real-time, ASR→MT→first TTS chunk ≈ 0.85 s |
| usable | M-series Pro/Max (non-Ultra) | est. RTF 0.6–1.0 (**unmeasured**) | ASR/MT real-time; TTS near real-time |
| runs | M2 / M3 base, 16 GB | RTF 1.48 streaming, first audio 323–510 ms | ASR/MT real-time; TTS produces audio slower than playback — streaming stutters |

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

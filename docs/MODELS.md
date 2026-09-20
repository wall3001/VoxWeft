# Models

Nothing is bundled. Default layout `~/.voxweft/models/` (`VOXWEFT_MODEL_DIR`).

| slot | dir | source | notes |
|---|---|---|---|
| TTS | `VoxCPM2/` | [OpenBMB/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | run `scripts/convert_vae.py` once (audiovae.pth → `vae_mlx.safetensors`); LM loads by name, no conversion |
| ASR | `Qwen3-ASR-1.7B-8bit/` | mlx-community/Qwen3-ASR-1.7B-8bit | 2.5 GB, RTF 0.11; transcripts identical to bf16 on our set. bf16 `Qwen/Qwen3-ASR-1.7B` if you insist (4.1 GB) |
| ASR alt | `whisper-large-v3-turbo/` | mlx-community | non-SEA languages |
| LLM/MT | `Qwen3-4B-Instruct-2507-4bit/` | mlx-community | `VOXWEFT_LLM_DIR`; 8B if you have the memory |
| MT alt | `Hy-MT2-1.8B-4bit/` | mlx-community | zh↔vi/th/km/my/en, fast |
| MT fallback | `nllb-200-1.3B/` | facebook | **CC-BY-NC-4.0** — research only; `pip install "voxweft[nllb]"` |

If Hugging Face is slow or blocked where you are, ModelScope mirrors the
Qwen and mlx-community repos; the sherpa-onnx Omnilingual ASR models are on
GitHub releases. Anonymous ModelScope downloads get rate-limited to
~0.1 MB/s after a quota — log in.

## Golden files for `tests/test_core_golden.py`

`vae_golden.npz`, `lm_golden.npz`, `enc_golden.npz` were produced by running
the official PyTorch implementation on fixed inputs. Drop them into the model
dir; the tests skip if absent. Regeneration scripts are not in this repo yet.

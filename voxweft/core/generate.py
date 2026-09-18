"""VoxCPM2 MLX inference: tokenizer, weight loading, autoregressive loop,
incremental VAE decode. Single-sequence (no batching) — edge deployment.
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .model import VoxCPM2MLX, VoxCPM2MLXConfig
from .vae import (AudioVAEDecoder, StreamingVAEDecoder, StreamingVAEState,
                  load_encoder_weights)

AUDIO_START = 101
REF_AUDIO_START = 103
REF_AUDIO_END = 104
N_DECODE_PAD = 12  # frames of latent context for incremental VAE decode


def load_tokenizer(model_path: str):
    """Llama tokenizer with multi-char Chinese tokens split to chars."""
    from transformers import LlamaTokenizerFast

    tok = LlamaTokenizerFast.from_pretrained(model_path)
    multichar = {t for t in tok.vocab
                 if len(t) >= 2 and all("一" <= c <= "鿿" for c in t)}

    def encode(text: str) -> list[int]:
        tokens = []
        for t in tok.tokenize(text):
            clean = t.replace("▁", "")
            if clean in multichar:
                tokens.extend(list(clean))
            else:
                tokens.append(t)
        return tok.convert_tokens_to_ids(tokens)

    return encode


def _fade_in(chunk: np.ndarray, ms: float, sr: int = 48000) -> np.ndarray:
    """Raised-cosine ramp over the first ``ms`` milliseconds. Copies; never
    touches the caller's array."""
    n = min(len(chunk), int(sr * ms / 1000.0))
    if n <= 0:
        return chunk
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32))
    out = chunk.astype(np.float32, copy=True)
    out[:n] *= ramp
    return out


class VoxCPM2Pipeline:
    def __init__(self, model_path: str, inference_timesteps: int = 10,
                 quantize_lm: int | None = None):
        p = Path(model_path)
        cfg_json = json.loads((p / "config.json").read_text())
        self.cfg = VoxCPM2MLXConfig.from_json(cfg_json)
        self.cfg.inference_timesteps = inference_timesteps

        self.model = VoxCPM2MLX(self.cfg)
        weights = mx.load(str(p / "model.safetensors"))
        self.model.load_weights(list(weights.items()))
        if quantize_lm:
            import mlx.nn as nn
            # LM stacks hold ~85% of weights; LocEnc/LocDiT stay bf16
            # (small + numerically sensitive in the CFM loop)
            nn.quantize(self.model.base_lm, bits=quantize_lm, group_size=64)
            nn.quantize(self.model.residual_lm, bits=quantize_lm, group_size=64)
        mx.eval(self.model.parameters())

        self.vae = AudioVAEDecoder()
        vae_weights = {k: v for k, v in mx.load(str(p / "vae_mlx.safetensors")).items()
                       if not k.startswith("encoder_raw.")}
        self.vae.load_weights(list(vae_weights.items()))
        mx.eval(self.vae.parameters())

        self.encode_text = load_tokenizer(str(p))
        self.patch = self.cfg.patch_size
        self.feat_dim = self.cfg.feat_dim
        self._vae_path = str(p / "vae_mlx.safetensors")
        self._encoder = None

    def encode_reference(self, wav_16k: np.ndarray) -> np.ndarray:
        """Reference wav (float32 mono @16kHz) → latents [T, 64] for cloning.
        Right-pads to patch*640=2560 alignment (official isolation mode)."""
        if self._encoder is None:
            self._encoder = load_encoder_weights(self._vae_path)
        align = self.patch * self._encoder.chunk
        if len(wav_16k) % align:
            wav_16k = np.pad(wav_16k, (0, align - len(wav_16k) % align))
        mu = self._encoder(mx.array(wav_16k[None].astype(np.float32)))
        mx.eval(mu)
        return np.array(mu[0])                      # frame-major [T, 64]

    def generate_streaming(self, text: str, ref_latents: np.ndarray | None = None,
                           temperature: float = 1.0, cfg_value: float = 2.0,
                           max_patches: int = 500,
                           warmup_patches: int = 0,
                           fade_in_ms: float = 8.0):
        """Yields (audio_chunk float32 @48kHz, patch_index) as patches are
        generated — stateful VAE stream, no boundary artifacts.

        Onset handling (the "first-frame pop") — see DEVLOG §2.3 for the
        measurements behind these defaults:

        * ``fade_in_ms`` — raised-cosine ramp on the first *emitted* chunk.
          Removes any step at playback start regardless of where the stream
          begins. 8 ms @ 48 kHz = 384 samples, inaudible as a fade. Default on.
        * ``warmup_patches`` — generate but do not emit the first N patches.
          mlx-audio does this (N=2) on the theory that early patches carry an
          onset transient. We measured six runs (zero-shot and clone) and found
          no such transient — patch 1 is usually *quieter* than the rest —
          while skipping two patches on a base M2 costs ~470 ms of first-audio
          latency (each patch is ~230 ms wall time at RTF 1.5) and *creates* a
          step by starting playback mid-speech. Default 0. Kept as a knob.
        """
        m = self.model
        m.reset_caches()
        text_tokens, feats, masks = self._build_prompt(text, ref_latents)

        latent, stop = m.step(mx.array(text_tokens), mx.array(feats),
                              mx.array(masks), temperature, cfg_value)
        mx.eval(latent)

        stream = StreamingVAEDecoder(self.vae)
        st = StreamingVAEState()
        n = 0
        min_patches = 3
        emitted = 0
        while True:
            n += 1
            frames = np.array(latent).T                  # [patch, 64]
            wav = stream.decode_chunk(mx.array(frames[None]), st)
            mx.eval(wav)
            if n > warmup_patches:
                chunk = np.array(wav[0])
                if emitted == 0 and fade_in_ms > 0:
                    chunk = _fade_in(chunk, fade_in_ms)
                emitted += 1
                yield chunk, n

            if (stop and n > min_patches) or n >= max_patches:
                return
            latent, stop = m.step(mx.array([0]), mx.array(frames[None]),
                                  mx.array([True]), temperature, cfg_value)
            mx.eval(latent)

    def _build_prompt(self, text: str, ref_latents: np.ndarray | None):
        text_tokens = self.encode_text(text) + [AUDIO_START]
        feats = np.zeros((len(text_tokens), self.patch, self.feat_dim), np.float32)
        masks = [False] * len(text_tokens)
        if ref_latents is not None:
            ref = ref_latents.reshape(-1, self.patch, self.feat_dim)
            pad = np.zeros((1, self.patch, self.feat_dim), np.float32)
            feats = np.concatenate([pad, ref, pad, feats], axis=0)
            text_tokens = ([REF_AUDIO_START] + [0] * ref.shape[0]
                           + [REF_AUDIO_END] + text_tokens)
            masks = [False] + [True] * ref.shape[0] + [False] + masks
        return text_tokens, feats, masks

    def generate(self, text: str, ref_latents: np.ndarray | None = None,
                 temperature: float = 1.0, cfg_value: float = 2.0,
                 max_patches: int = 500, verbose: bool = False):
        """→ (waveform float32 @48kHz, stats dict).

        ref_latents: optional [T*patch, 64] reference-voice latents
        (isolation-mode cloning), from AudioVAE encoder.
        """
        m = self.model
        m.reset_caches()

        text_tokens = self.encode_text(text) + [AUDIO_START]
        feats = np.zeros((len(text_tokens), self.patch, self.feat_dim), np.float32)
        masks = [False] * len(text_tokens)

        if ref_latents is not None:
            ref = ref_latents.reshape(-1, self.patch, self.feat_dim)
            pad = np.zeros((1, self.patch, self.feat_dim), np.float32)
            feats = np.concatenate([pad, ref, pad, feats], axis=0)
            text_tokens = ([REF_AUDIO_START] + [0] * ref.shape[0]
                           + [REF_AUDIO_END] + text_tokens)
            masks = [False] + [True] * ref.shape[0] + [False] + masks

        t0 = time.perf_counter()
        latent, stop = m.step(
            mx.array(text_tokens),
            mx.array(feats),
            mx.array(masks),
            temperature, cfg_value,
        )
        mx.eval(latent)
        t_prefill = time.perf_counter() - t0

        # autoregressive patch loop; full-sequence VAE decode at the end.
        # latents stay on device (no per-step GPU↔CPU round-trip).
        frames_list = []
        n = 0
        min_patches = 3            # official min_len=2 gate: no stop while i<=2
        while True:
            n += 1
            frames = mx.transpose(latent)                # [patch, 64] on device
            frames_list.append(frames)

            if stop and n > min_patches:
                break
            if n >= max_patches:
                break
            latent, stop = m.step(
                mx.array([0]),
                frames[None],                            # [1, patch, 64] on device
                mx.array([True]),
                temperature, cfg_value,
            )
            if verbose and n % 10 == 0:
                print(f"  patch {n}, {(time.perf_counter()-t0)*1000:.0f}ms")

        latents = mx.concatenate(frames_list, axis=0)    # [n*patch, 64] on device
        wav = self.vae(latents[None])
        mx.eval(wav)
        waveform = np.array(wav[0])                      # one device→host copy
        t_first_audio = time.perf_counter() - t0
        wall = time.perf_counter() - t0
        stats = {
            "patches": n,
            "audio_s": len(waveform) / 48000,
            "wall_s": wall,
            "rtf": wall / (len(waveform) / 48000),
            "prefill_s": t_prefill,
            "first_audio_s": t_first_audio,
        }
        return waveform, stats

"""Golden alignment tests. Set VOXCPM2_MODEL to the model directory containing
model.safetensors, vae_mlx.safetensors and the golden .npz files
(vae_golden.npz / lm_golden.npz / enc_golden.npz, produced by scripts against
the official PyTorch implementation)."""
import os
import sys
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX (Apple Silicon) required")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
MODEL = os.environ.get("VOXCPM2_MODEL", os.path.expanduser("~/.voxweft/models/VoxCPM2"))
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL, "vae_golden.npz")),
    reason=f"golden files not found under {MODEL} (see docs/MODELS.md)")


def test_vae_decoder_golden():
    from voxweft.core.vae import AudioVAEDecoder
    g = np.load(f"{MODEL}/vae_golden.npz")
    vae = AudioVAEDecoder()
    w = {k: v for k, v in mx.load(f"{MODEL}/vae_mlx.safetensors").items()
         if not k.startswith("encoder_raw.")}
    vae.load_weights(list(w.items()))
    out = np.array(vae(mx.array(g["z"].transpose(0, 2, 1)))[0])
    assert np.corrcoef(out, g["wav"][0])[0, 1] > 0.999


def test_streaming_equals_full():
    from voxweft.core.vae import (AudioVAEDecoder, StreamingVAEDecoder,
                                 StreamingVAEState)
    g = np.load(f"{MODEL}/vae_golden.npz")
    z = g["z"].transpose(0, 2, 1)
    vae = AudioVAEDecoder()
    w = {k: v for k, v in mx.load(f"{MODEL}/vae_mlx.safetensors").items()
         if not k.startswith("encoder_raw.")}
    vae.load_weights(list(w.items()))
    full = np.array(vae(mx.array(z))[0])
    stream, st = StreamingVAEDecoder(vae), StreamingVAEState()
    chunks = [np.array(stream.decode_chunk(mx.array(z[:, i:i+4, :]), st)[0])
              for i in range(0, z.shape[1], 4)]
    sv = np.concatenate(chunks)
    assert np.corrcoef(full, sv)[0, 1] > 0.9999


if __name__ == "__main__":
    test_vae_decoder_golden(); print("vae decoder golden PASS")
    test_streaming_equals_full(); print("streaming==full PASS")

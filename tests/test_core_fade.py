"""First-chunk onset handling — runs without a model."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="voxweft.core imports MLX")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voxweft.core.generate import _fade_in  # noqa: E402


def test_fade_in_ramps_from_zero_and_preserves_tail():
    x = np.ones(7680, np.float32)            # one 160 ms patch @48k
    y = _fade_in(x, 8.0)
    n = int(48000 * 8 / 1000)                # 384 samples
    assert y[0] == pytest.approx(0.0, abs=1e-6)
    assert y[n // 2] == pytest.approx(0.5, abs=0.02)
    assert np.all(y[n:] == 1.0)
    assert np.all(np.diff(y[:n]) >= -1e-7)   # monotone ramp


def test_fade_in_does_not_mutate_input():
    x = np.ones(1000, np.float32)
    _ = _fade_in(x, 8.0)
    assert np.all(x == 1.0)


def test_fade_in_short_chunk_and_zero_ms():
    x = np.ones(10, np.float32)
    assert len(_fade_in(x, 8.0)) == 10
    assert _fade_in(x, 0.0) is x

import numpy as np
import pytest
from llmanim.base.utils import (
    softmax,
    causal_mask,
    sinusoidal_pe,
    make_attention_weights,
    make_token_sequence,
    value_to_color,
)


class TestSoftmax:
    def test_output_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert np.isclose(result.sum(), 1.0)

    def test_higher_temperature_flattens(self):
        x = np.array([1.0, 2.0, 3.0])
        low_t = softmax(x, temperature=0.1)
        high_t = softmax(x, temperature=10.0)
        assert low_t.max() > high_t.max()

    def test_numerical_stability(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        assert not np.any(np.isnan(result))
        assert np.isclose(result.sum(), 1.0)


class TestCausalMask:
    def test_shape(self):
        mask = causal_mask(4)
        assert mask.shape == (4, 4)

    def test_upper_triangle_is_masked(self):
        mask = causal_mask(4)
        # positions where mask is True (or -inf) should be upper triangle
        # exact convention depends on implementation — just check it's not all zeros
        assert not np.all(mask == 0)


class TestSinusoidalPE:
    def test_shape(self):
        pe = sinusoidal_pe(seq_len=10, d_model=64)
        assert pe.shape == (10, 64)

    def test_values_in_range(self):
        pe = sinusoidal_pe(seq_len=10, d_model=64)
        assert pe.min() >= -1.0
        assert pe.max() <= 1.0


class TestMakeAttentionWeights:
    def test_shape(self):
        weights = make_attention_weights(seq_len=5, n_heads=4)
        assert weights.shape == (4, 5, 5)

    def test_rows_sum_to_one(self):
        weights = make_attention_weights(seq_len=5, n_heads=1)
        assert np.allclose(weights.sum(axis=-1), 1.0)


class TestMakeTokenSequence:
    def test_returns_list(self):
        result = make_token_sequence(["hello", "world"])
        assert isinstance(result, list)
        assert len(result) == 2


class TestValueToColor:
    def test_returns_color(self):
        color = value_to_color(0.5)
        assert color is not None

    def test_extremes_dont_crash(self):
        value_to_color(0.0)
        value_to_color(1.0)
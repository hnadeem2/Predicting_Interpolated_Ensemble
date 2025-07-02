import numpy as np
import pytest
from pie.sequence.sequence_utils import combine_features_from_indices, get_max_likelihood_seq


def test_combine_features_uniform_weights():
    # Two arrays of shape (5, 21) with ones and zeros
    arr1 = np.ones((5, 21))
    arr2 = np.zeros((5, 21))

    # Map all ref positions to seq positions 0–4
    index_maps = [list(range(5)), list(range(5))]

    output = combine_features_from_indices(
        [arr1, arr2],
        index_maps,
        ref_len=5
    )

    # Should average to 0.5 for all positions
    expected = np.ones((5, 21)) * 0.5
    np.testing.assert_allclose(output, expected)


def test_combine_features_with_weights_and_missing_indices():
    arr1 = np.ones((3, 21))  # All 1s
    arr2 = np.zeros((3, 21))  # All 0s

    index_maps = [
        [0, 1, None],  # arr1 missing last position
        [0, None, 2]   # arr2 missing middle position
    ]
    weights = [0.3, 0.7]

    result = combine_features_from_indices([arr1, arr2], index_maps, ref_len=3, weights=weights)

    # First position: weighted average of 1*0.3 + 0*0.7 = 0.3 / 1.0 = 0.3
    # Second: 1*0.3 = 0.3 / 0.3 = 1.0
    # Third: 0*0.7 = 0 / 0.7 = 0.0
    expected = np.array([
        [0.3] * 21,
        [1.0] * 21,
        [0.0] * 21
    ])
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_combine_features_asserts_on_bad_weights():
    arrs = [np.ones((3, 21)), np.zeros((3, 21))]
    idx_maps = [[0, 1, 2], [0, 1, 2]]
    with pytest.raises(AssertionError, match="Weights length must match"):
        combine_features_from_indices(arrs, idx_maps, 3, weights=[1.0])  # wrong length

    with pytest.raises(AssertionError, match="Weights must sum to 1"):
        combine_features_from_indices(arrs, idx_maps, 3, weights=[0.2, 0.2])  # sum != 1


def test_get_max_likelihood_seq_correct():
    prob = np.eye(3, 21)  # One-hot for 3 positions
    alphabet = [chr(ord('A') + i) for i in range(21)]  # A-U

    expected_seq = ''.join(alphabet[i] for i in range(3))
    result = get_max_likelihood_seq(prob, alphabet)
    assert result == expected_seq


def test_get_max_likelihood_seq_shape_mismatch():
    prob = np.eye(3, 20)  # 20 columns, not 21
    alphabet = [chr(ord('A') + i) for i in range(21)]
    with pytest.raises(ValueError, match="Shape mismatch"):
        get_max_likelihood_seq(prob, alphabet)

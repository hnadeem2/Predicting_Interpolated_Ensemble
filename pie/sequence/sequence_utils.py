import numpy as np
from typing import List


def combine_features_from_indices(arrays, index_maps, ref_len, weights=None):
    """
    Combine features from multiple arrays using precomputed alignment indices,
    computing a weighted average at each position.

    Parameters:
        arrays: list of [L_i x 21] arrays
        index_maps: list of [ref_len] index lists mapping reference to sequence indices
        ref_len: int, length of the reference sequence
        weights: list or np.array of floats, same length as arrays, summing to 1.
                 If None, uniform weights are used.

    Returns:
        output: np.array of shape [ref_len x 21]
    """
    if weights is None:
        weights = np.ones(len(arrays)) / len(arrays)
    else:
        weights = np.array(weights)
        assert len(weights) == len(arrays), "Weights length must match number of arrays."
        assert np.isclose(weights.sum(), 1.0), "Weights must sum to 1."

    output = np.zeros((ref_len, 21), dtype=np.float32)

    for i in range(ref_len):
        weighted_vectors = []
        weighted_coeffs = []
        for arr_idx, (arr, idx_map) in enumerate(zip(arrays, index_maps)):
            seq_pos = idx_map[i]
            if seq_pos is not None:
                weighted_vectors.append(arr[seq_pos] * weights[arr_idx])
                weighted_coeffs.append(weights[arr_idx])
        if weighted_vectors:
            # Normalize weights if some are missing at this position
            wsum = sum(weighted_coeffs)
            output[i] = sum(weighted_vectors) / wsum

    return output


def get_max_likelihood_seq(prob_arr: np.ndarray, alphabet: List[str]) -> str:
    """
    Given an array of shape (L, 21) with categorical distributions and an
    alphabet of 21 letters, return the maximum likelihood sequence.
    
    Parameters:
        prob_arr: np.ndarray of shape (L, 21)
        alphabet: List of 21 single-character strings

    Returns:
        A string of length L representing the max likelihood sequence
    """
    if prob_arr.shape[1] != len(alphabet):
        raise ValueError(f"Shape mismatch: prob_arr has shape {prob_arr.shape}, but alphabet has length {len(alphabet)}")

    max_indices = np.argmax(prob_arr, axis=1)  # shape: (L,)
    return ''.join(alphabet[i] for i in max_indices)
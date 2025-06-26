'''Functionality to handle probability mixtures.
'''

import os
import subprocess
import numpy as np
from Bio.Align import PairwiseAligner
from itertools import product, combinations
import mdtraj as md 
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import shortest_path


def run_pmpnn(
    pdb_path,
    output_dir,
    mpnn_path,
    pmpnn_script,
    seed=10, 
    temp=0.1, 
    batch_size=1
    ):
    """
    Runs ProteinMPNN on a given PDB structure.

    Args:
        pdb_path (str): Path to input PDB file.
        output_dir (str): Directory for saving outputs.
        mpnn_path (str): Path to the ProteinMPNN run script.
        pmpnn_template (str, optional): Path to the ProteinMPNN shell script template. Defaults to "pmpnn_template.sh".
        seed (int, optional): Random seed for reproducibility. Defaults to 10.
        temp (float, optional): Sampling temperature. Defaults to 0.1.
        batch_size (int, optional): Batch size for generation. Defaults to 1.

    Returns:
        Tuple[str, str]: Paths to the generated FASTA sequence file and NPZ probability file.
    """
    script_path = pmpnn_script
    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]

    subprocess.run([
        'bash',
        script_path,
        pdb_path,
        output_dir,
        str(temp),
        str(seed),
        str(batch_size),
        mpnn_path
    ], 
    check=True)

    return f"{output_dir}/seqs/{pdb_name}.fa", f"{output_dir}/probs/{pdb_name}.npz"



def compute_alignment_indices(sequences, ref_seq):
    """
    Align multiple sequences to a reference sequence.
    Returns a list of index maps: ref_pos → seq_pos (or None for gaps).
    """
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.target_open_gap_score = -10
    aligner.target_extend_gap_score = -10
    aligner.query_open_gap_score = -1
    aligner.query_extend_gap_score = -0.1

    index_maps = []

    for seq in sequences:
        alignment = aligner.align(ref_seq, seq)[0]
        ref_to_seq_idx = [None] * len(ref_seq)

        ref_pos = 0
        seq_pos = 0
        for (ref_start, ref_end), (seq_start, seq_end) in zip(*alignment.aligned):
            while ref_pos < ref_start:
                ref_to_seq_idx[ref_pos] = None
                ref_pos += 1
            for _ in range(ref_end - ref_start):
                ref_to_seq_idx[ref_pos] = seq_pos
                ref_pos += 1
                seq_pos += 1

        index_maps.append(ref_to_seq_idx)

    return index_maps


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


def load_cif_trajs(filepaths):
    """
    Load .cif files into separate md.Trajectory objects.

    Parameters:
        filepaths (list of str): List of paths to .cif files.

    Returns:
        list of md.Trajectory: Loaded trajectories.
    """
    trajs = [md.load_cif(fp) for fp in filepaths]
    return trajs


def compute_pairwise_fape(trajs, fape_fn):
    """
    Compute all pairwise FAPE scores between trajectories.

    Parameters:
        trajs (list of md.Trajectory): List of trajectories.
        fape_fn (callable): Function that computes FAPE between two trajs.

    Returns:
        np.ndarray: Symmetric matrix of shape (N, N) with FAPE scores.
    """
    N = len(trajs)
    fape_matrix = np.zeros((N, N))
    for i, j in combinations(range(N), 2):
        score = fape_fn(trajs[i], trajs[j])
        fape_matrix[i, j] = score
        fape_matrix[j, i] = score
    return fape_matrix


def shortest_fape_path(fape_matrix, source_idx, target_idx):
    """
    Compute shortest path and largest gap in FAPE distance graph.

    Parameters:
        fape_matrix (np.ndarray): Pairwise FAPE score matrix.
        source_idx (int): Index of source structure.
        target_idx (int): Index of target structure.

    Returns:
        path (list of int): Indices of structures in shortest path.
        gap_indices (tuple of int): Pair of indices with largest FAPE in path.
    """
    dist_matrix = fape_matrix.copy()
    np.fill_diagonal(dist_matrix, 0)
    _, predecessors = shortest_path(dist_matrix, directed=False, return_predecessors=True)
    
    # Reconstruct shortest path
    path = []
    current = target_idx
    while current != source_idx:
        path.append(current)
        current = predecessors[source_idx, current]
    path.append(source_idx)
    path = path[::-1]

    # Find largest gap
    max_gap = -np.inf
    gap_pair = (None, None)
    for a, b in zip(path[:-1], path[1:]):
        gap = fape_matrix[a, b]
        if gap > max_gap:
            max_gap = gap
            gap_pair = (a, b)

    return path, gap_pair


def interpolate_arrays(arr_a, arr_b, step=0.1):
    """
    Linearly interpolate between two arrays with given step.

    Parameters:
        arr_a (np.ndarray): Start array, shape (L, 21).
        arr_b (np.ndarray): End array, shape (L, 21).
        step (float): Step size between 0 and 1.

    Returns:
        list of np.ndarray: Interpolated arrays including arr_a and arr_b.
    """
    assert arr_a.shape == arr_b.shape, "Arrays must have the same shape"
    t_values = np.arange(0, 1 + step, step)
    interpolated = [(1 - t) * arr_a + t * arr_b for t in t_values]
    return interpolated


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

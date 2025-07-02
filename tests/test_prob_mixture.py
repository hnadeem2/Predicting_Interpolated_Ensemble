import numpy as np
import pytest
from pathlib import Path
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Align import MultipleSeqAlignment
from Bio import AlignIO

from pie.predict_utils import (
    compute_alignment_indices,
    read_alignment_indices,
    combine_features_from_indices,
    get_max_likelihood_seq,
)

# ----------------------
# compute_alignment_indices
# ----------------------

def test_compute_alignment_indices_basic():
    ref = "ACDEFGHIK"
    seqs = ["ACDEFGHIK", "ACDFFGHIK"]
    result = compute_alignment_indices(seqs, ref)

    for idx_map in result:
        assert len(idx_map) == len(ref)
        assert all(i is None or isinstance(i, int) for i in idx_map)

# ----------------------
# read_alignment_indices
# ----------------------

def test_read_alignment_indices(tmp_path):
    aln_file = tmp_path / "aln.fa"
    ref_seq = "ACDEFGHIK"

    alignment = MultipleSeqAlignment([
        SeqRecord(Seq("ACDEFGHIK"), id="ref"),
        SeqRecord(Seq("A--EFGHIK"), id="mut1"),
        SeqRecord(Seq("ACDEFGHIK"), id="mut2")
    ])
    AlignIO.write(alignment, aln_file, "fasta")

    result = read_alignment_indices(aln_file, ref_seq)
    assert len(result) == 3
    for idx_map in result:
        assert len(idx_map) == len(ref_seq)
        assert all(i is None or isinstance(i, int) for i in idx_map)

def test_read_alignment_indices_ref_not_found(tmp_path):
    aln_file = tmp_path / "aln_missing_ref.fa"
    ref_seq = "ACDEFGHIK"

    alignment = MultipleSeqAlignment([
        SeqRecord(Seq("AXDEFGHIK"), id="wrong"),
        SeqRecord(Seq("A--EFGHIK"), id="mut1"),
    ])
    AlignIO.write(alignment, aln_file, "fasta")

    with pytest.raises(ValueError, match="Reference sequence not found"):
        read_alignment_indices(aln_file, ref_seq)

# ----------------------
# combine_features_from_indices
# ----------------------

def test_combine_features_uniform_weights():
    ref_len = 3
    A = np.eye(3, 21)
    B = np.roll(A, shift=1, axis=1)
    arrays = [A, B]
    index_maps = [[0, 1, 2], [0, 1, 2]]

    combined = combine_features_from_indices(arrays, index_maps, ref_len)
    assert combined.shape == (3, 21)
    np.testing.assert_allclose(combined.sum(axis=1), np.ones(3), rtol=1e-5)

def test_combine_features_partial_alignment():
    ref_len = 3
    A = np.eye(3, 21)
    B = np.roll(A, shift=1, axis=1)
    arrays = [A, B]
    index_maps = [[0, 1, 2], [None, 1, 2]]

    combined = combine_features_from_indices(arrays, index_maps, ref_len)
    assert combined.shape == (3, 21)
    assert np.allclose(combined[0].sum(), 1.0)

def test_combine_features_with_weights():
    ref_len = 2
    A = np.full((2, 21), 0.2)
    B = np.full((2, 21), 0.8)
    arrays = [A, B]
    index_maps = [[0, 1], [0, 1]]
    weights = [0.25, 0.75]

    result = combine_features_from_indices(arrays, index_maps, ref_len, weights)
    expected = (A * 0.25 + B * 0.75)
    np.testing.assert_allclose(result, expected)

def test_combine_features_weight_error():
    A = np.random.rand(2, 21)
    B = np.random.rand(2, 21)
    with pytest.raises(AssertionError, match="Weights must sum to 1"):
        combine_features_from_indices([A, B], [[0, 1], [0, 1]], 2, weights=[0.5, 0.3])

# ----------------------
# get_max_likelihood_seq
# ----------------------

def test_get_max_likelihood_seq_basic():
    arr = np.zeros((3, 21))
    arr[0, 0] = 1.0
    arr[1, 1] = 1.0
    arr[2, 2] = 1.0
    alphabet = list("ACDEFGHIKLMNPQRSTVWYX")

    result = get_max_likelihood_seq(arr, alphabet)
    assert result == "ACD"

def test_get_max_likelihood_seq_shape_mismatch():
    arr = np.random.rand(5, 20)
    with pytest.raises(ValueError, match="Shape mismatch"):
        get_max_likelihood_seq(arr, list("ACDEFGHIKLMNPQRSTVWYX"))
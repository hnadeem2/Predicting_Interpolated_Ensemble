'''Functionality to handle user input.
'''

from Bio import pairwise2
import numpy as np

def align_to_reference(seq, ref):
    """
    Aligns `seq` to `ref` using asymmetric gap penalties
    (gaps in reference heavily penalized, gaps in seq lightly penalized).
    Returns a list of indices from ref to seq (None if aligned to gap).
    """
    alignments = pairwise2.align.globalmd(
        ref, seq,
        match=2, mismatch=-1,
        open_A=-10, extend_A=-10,  # gap in ref (target): high penalty
        open_B=-1, extend_B=-0.1   # gap in seq (query): normal penalty
    )
    
    ref_aln, seq_aln, _, _, _ = alignments[0]
    
    ref_to_seq_idx = []
    seq_i = 0
    for r, s in zip(ref_aln, seq_aln):
        if r != '-':
            if s != '-':
                ref_to_seq_idx.append(seq_i)
            else:
                ref_to_seq_idx.append(None)
        if s != '-':
            seq_i += 1
    return ref_to_seq_idx


def combine_features(A, seq_A, B, seq_B, ref_seq):
	"""Combination of probabilities distributions from A and B for first round.
	"""
    O = len(ref_seq)
    output = np.zeros((O, 21), dtype=np.float32)

    idx_A = align_to_reference(seq_A, ref_seq)
    idx_B = align_to_reference(seq_B, ref_seq)

    for i in range(O):
        vecs = []
        if idx_A[i] is not None:
            vecs.append(A[idx_A[i]])
        if idx_B[i] is not None:
            vecs.append(B[idx_B[i]])
        if vecs:
            output[i] = np.mean(vecs, axis=0)
    return output
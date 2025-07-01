from Bio.Align import PairwiseAligner
from Bio import AlignIO


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


def read_alignment_indices(aln_file, ref_seq):
    """
    Read alignment from FASTA file.
    """
    msa = AlignIO.read(aln_file, "fasta")

    # Find the reference sequence in the alignment
    ref_record = next((rec for rec in msa if rec.seq.ungap("-") == ref_seq), None)
    if ref_record is None:
        raise ValueError("Reference sequence not found in alignment file.")

    ref_aln = str(ref_record.seq)
    alignment_indices = []

    for rec in msa:
        seq_aln = str(rec.seq)
        ref_to_seq = []
        ref_pos, seq_pos = 0, 0
        for r, s in zip(ref_aln, seq_aln):
            if r != '-' and s != '-':
                ref_to_seq.append(seq_pos)
            elif r != '-' and s == '-':
                ref_to_seq.append(None)
            if r != '-':
                ref_pos += 1
            if s != '-':
                seq_pos += 1
        if len(ref_to_seq) != len(ref_seq):
            raise ValueError(f"Aligned reference length ({len(ref_to_seq)}) does not match unaligned ref_seq length ({len(ref_seq)})")
        alignment_indices.append(ref_to_seq)

    return alignment_indices
from Bio.Align import PairwiseAligner
from Bio import AlignIO


def compute_alignment_indices(sequences, ref_seq):
    """
    Align multiple sequences to a reference sequence.

    Returns:
        - aligned_seqs: list[str]  # each sequence aligned to ref_seq (with gaps)
        - index_maps:  list[list[Optional[int]]]  # ref_pos -> seq_pos (None if gap)
    """
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.target_open_gap_score = -10
    aligner.target_extend_gap_score = -10
    aligner.query_open_gap_score = -1
    aligner.query_extend_gap_score = -0.1

    aligned_seqs = []
    index_maps = []

    for seq in sequences:
        aln = aligner.align(ref_seq, seq)[0]

        # Reconstruct aligned strings from block coordinates
        ref_aln_chars = []
        seq_aln_chars = []
        ref_pos = 0
        seq_pos = 0
        ref_blocks, seq_blocks = aln.aligned

        for (r_start, r_end), (s_start, s_end) in zip(ref_blocks, seq_blocks):
            # gaps in ref before this block
            if r_start > ref_pos:
                ref_aln_chars.extend(ref_seq[ref_pos:r_start])
                seq_aln_chars.extend('-' * (r_start - ref_pos))
            # gaps in seq before this block
            if s_start > seq_pos:
                ref_aln_chars.extend('-' * (s_start - seq_pos))
                seq_aln_chars.extend(seq[seq_pos:s_start])

            # aligned block
            ref_aln_chars.extend(ref_seq[r_start:r_end])
            seq_aln_chars.extend(seq[s_start:s_end])

            ref_pos = r_end
            seq_pos = s_end

        # trailing tails
        if ref_pos < len(ref_seq):
            ref_aln_chars.extend(ref_seq[ref_pos:])
            seq_aln_chars.extend('-' * (len(ref_seq) - ref_pos))
        if seq_pos < len(seq):
            ref_aln_chars.extend('-' * (len(seq) - seq_pos))
            seq_aln_chars.extend(seq[seq_pos:])

        ref_aln = ''.join(ref_aln_chars)
        seq_aln = ''.join(seq_aln_chars)

        # Build ref_pos -> seq_pos map (0-based), advancing seq index across insertions
        ref_to_seq_idx = []
        seq_non_gap_idx = 0
        for r, s in zip(ref_aln, seq_aln):
            if r != '-':
                if s == '-':
                    ref_to_seq_idx.append(None)
                else:
                    ref_to_seq_idx.append(seq_non_gap_idx)
            if s != '-':
                seq_non_gap_idx += 1

        # Sanity: map length matches ungapped ref length
        if len(ref_to_seq_idx) != len(ref_seq):
            raise RuntimeError(
                f"Internal error: map length {len(ref_to_seq_idx)} != len(ref_seq) {len(ref_seq)}"
            )

        aligned_seqs.append(seq_aln)
        index_maps.append(ref_to_seq_idx)

    return aligned_seqs, index_maps


def read_alignment_indices(aln_file, ref_seq):
    """
    Read alignment from FASTA file.
    """
    msa = AlignIO.read(aln_file, "fasta")

    # Find the reference sequence in the alignment
    ref_record = next((rec for rec in msa if rec.seq.replace("-", "") == ref_seq), None)
    if ref_record is None:
        raise ValueError("Reference sequence not found in alignment file.")

    ref_aln = str(ref_record.seq)
    aligned_seqs = []
    alignment_indices = []

    for rec in msa:
        seq_aln = str(rec.seq)
        aligned_seqs.append(seq_aln)
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

    return aligned_seqs[1:], alignment_indices[1:] # Exclude ref
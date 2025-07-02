import pytest
from pie.sequence.align import compute_alignment_indices, read_alignment_indices
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Align import MultipleSeqAlignment
from Bio import AlignIO
from pathlib import Path


def test_compute_alignment_indices_basic():
    ref_seq = "ACDEFGHIK"
    seqs = [
        "ACDEFGHIK",
        "ACDEFGHIK",  # identical
        "ACD-FGH-K"   # with gaps, will be aligned
    ]
    results = compute_alignment_indices(seqs, ref_seq)

    for mapping in results:
        assert len(mapping) == len(ref_seq)
        assert all(isinstance(i, (int, type(None))) for i in mapping)


def test_compute_alignment_indices_with_gap():
    ref_seq = "ACDEF"
    seqs = ["ADEF"]

    result = compute_alignment_indices(seqs, ref_seq)
    assert len(result) == 1
    idx_map = result[0]
    # The C in ref_seq is missing from seqs[0], so should be None
    assert idx_map[0] == 0  # A
    assert idx_map[1] is None  # C
    assert idx_map[2] == 1  # D
    assert idx_map[3] == 2  # E
    assert idx_map[4] == 3  # F


def test_read_alignment_indices(tmp_path):
    ref_seq = "ACDEFGHIK"

    # Simulate aligned sequences with gaps
    aln_path = tmp_path / "example_alignment.fasta"
    records = [
        SeqRecord(Seq("A--CDEFGHIK"), id="ref"),
        SeqRecord(Seq("A--C-E-GHIK"), id="seq1")
    ]
    AlignIO.write(MultipleSeqAlignment(records), aln_path, "fasta")

    # Remove the gaps to match the unaligned `ref_seq`
    ref_seq_nogap = "ACDEFGHIK"

    idx_maps = read_alignment_indices(aln_path, ref_seq_nogap)

    assert len(idx_maps) == 2
    for idx_map in idx_maps:
        assert len(idx_map) == len(ref_seq_nogap)
        assert all(isinstance(i, (int, type(None))) for i in idx_map)


def test_read_alignment_indices_raises_on_missing_ref(tmp_path):
    aln_path = tmp_path / "bad_alignment.fasta"
    records = [
        SeqRecord(Seq("AXXXX"), id="not_ref"),
        SeqRecord(Seq("AYYYY"), id="seq1")
    ]
    AlignIO.write(MultipleSeqAlignment(records), aln_path, "fasta")

    with pytest.raises(ValueError, match="Reference sequence not found"):
        read_alignment_indices(aln_path, "ABCDE")

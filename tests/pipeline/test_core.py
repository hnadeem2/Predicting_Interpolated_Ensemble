import pytest
import numpy as np
from pathlib import Path
from biotite.structure import AtomArray, Atom
from pie.pipeline.core import load_modeled_seq, load_templates, find_anchors
from pie.data_structs import Structure


def make_mock_protein_chain(res_ids, res_names, chain_id="A"):
    atoms = []
    atom_arr = AtomArray(len(res_ids))
    for i, res_id in enumerate(res_ids):
        atom_arr[i] = Atom([0, 0, 0], chain_id=chain_id, res_id=res_id, res_name=res_names[i], atom_name="CA")
    
    return atom_arr


def test_load_modeled_seq(monkeypatch):
    mock_chain = make_mock_protein_chain(
        res_ids=[1, 2, 4],  # missing residue 3
        res_names=["ALA", "GLY", "LYS"]
    )

    monkeypatch.setattr("pie.pipeline.core.load_structure", lambda path: mock_chain)
    seq = load_modeled_seq("mock.pdb", chain_id="A")
    assert seq == "AG-K"


def test_load_templates(monkeypatch, tmp_path):
    fake_seq = "ACD"
    fake_npz = {"probs": np.ones((3, 21))}

    def mock_load_modeled_seq(pdb_path, chain_id):
        return fake_seq

    def mock_run_pmpnn(pdb_path, **kwargs):
        out_path = tmp_path / "mock_output.npz"
        np.savez(out_path, **fake_npz)
        return "fake.fasta", out_path

    monkeypatch.setattr("pie.pipeline.core.load_modeled_seq", mock_load_modeled_seq)
    monkeypatch.setattr("pie.pipeline.core.run_pmpnn", mock_run_pmpnn)
    monkeypatch.setattr("pie.pipeline.core.compute_alignment_indices", lambda seqs, ref: [[0, 1, 2] for _ in seqs])

    paths = [tmp_path / "1.pdb", tmp_path / "2.pdb"]
    for path in paths:
        path.write_text("MOCK")

    structs = load_templates(paths, "ACD", ["A", "A"], output_dir=tmp_path, mpnn_path=tmp_path, pmpnn_script="mock.sh")
    assert len(structs) == 2
    assert all(isinstance(s, Structure) for s in structs)


def test_find_anchors(monkeypatch):
    def mock_fape_fn(*args, **kwargs):
        return 1.0

    monkeypatch.setattr("pie.pipeline.core.fape_fn", mock_fape_fn)
    monkeypatch.setattr("pie.pipeline.core.compute_pairwise_fape", lambda s, f, m: np.array([[0, 1], [1, 0]]))
    monkeypatch.setattr("pie.pipeline.core.shortest_fape_path_mst", lambda mat, src, tgt: ([0, 1], (0, 1)))

    s1 = Structure(identity="a", structure_path=Path("a.pdb"), sequence="AAA", prob_dist=np.ones((3, 21)))
    s2 = Structure(identity="b", structure_path=Path("b.pdb"), sequence="BBB", prob_dist=np.ones((3, 21)))
    s1.aligned_indices = np.arange(3)
    s2.aligned_indices = np.arange(3)

    path, anchors, mat = find_anchors([s1, s2])
    assert path == [0, 1]
    assert len(anchors) == 2
    assert mat.shape == (2, 2)
import pytest
import numpy as np
from pathlib import Path
from pie.data_structs import Structure


def test_structure_valid_input(tmp_path):
    L = 10
    sequence = "A" * L
    prob_dist = np.random.rand(L, 21)
    prob_dist /= prob_dist.sum(axis=1, keepdims=True)

    struct = Structure(
        identity="test_struct",
        parents=[],
        parent_weights=[],
        structure_path=tmp_path / "dummy.pdb",
        sequence=sequence,
        prob_dist=prob_dist
    )

    assert struct.identity == "test_struct"
    assert struct.structure_path.name == "dummy.pdb"
    assert struct.aligned_indices.shape == (L,)
    assert np.all(struct.aligned_indices == np.arange(L))


def test_invalid_prob_dist_type():
    with pytest.raises(TypeError, match="prob_dist must be a numpy array"):
        Structure(
            identity="bad_type",
            parents=[],
            parent_weights=[],
            structure_path=Path("x.pdb"),
            sequence="AAAAA",
            prob_dist=[[0.05] * 21] * 5,  # not a numpy array
        )


def test_invalid_prob_dist_shape():
    prob_dist = np.random.rand(5, 20)  # Wrong shape

    with pytest.raises(ValueError, match="prob_dist must have shape"):
        Structure(
            identity="bad_shape",
            parents=[],
            parent_weights=[],
            structure_path=Path("x.pdb"),
            sequence="AAAAA",
            prob_dist=prob_dist,
        )


def test_empty_sequence_and_prob_dist():
    with pytest.raises(ValueError, match="prob_dist must have shape"):
        Structure(
            identity="empty",
            parents=[],
            parent_weights=[],
            structure_path=Path("empty.pdb"),
            sequence="",
            prob_dist=np.empty((0, 21)),
        )
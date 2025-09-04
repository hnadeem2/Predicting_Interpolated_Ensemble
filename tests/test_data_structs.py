import numpy as np
import pytest
from pathlib import Path
from pie.data_structs import Structure


def test_structure_valid_initialization():
    sequence = "ACDE"
    prob_dist = np.random.rand(4, 21)
    prob_dist /= prob_dist.sum(axis=1, keepdims=True)

    structure = Structure(
        identity="template_1",
        structure_path=Path("fake.pdb"),
        sequence=sequence,
        prob_dist=prob_dist
    )

    assert structure.identity == "template_1"
    assert structure.structure_path.name == "fake.pdb"
    assert structure.sequence == sequence
    assert structure.prob_dist.shape == (4, 21)
    assert np.array_equal(structure.aligned_indices, np.arange(4))


def test_structure_raises_if_prob_dist_not_array():
    with pytest.raises(TypeError, match="prob_dist must be a numpy array"):
        Structure(
            identity="bad_struct",
            structure_path=Path("file.pdb"),
            sequence="ACDE",
            prob_dist=[[0.05] * 21] * 4
        )


def test_structure_raises_if_prob_dist_wrong_shape():
    with pytest.raises(ValueError, match="prob_dist length"):
        Structure(
            identity="bad_struct",
            structure_path=Path("file.pdb"),
            sequence="ACDE--", # Gapless len = 4, gapped len = 6
            prob_dist=np.random.rand(5, 21)  # Should be shape (4, 21)
        )
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from pie.data_structs import Structure
from pie.fape_util import compute_pairwise_fape, shortest_fape_path_mst

# --- Fixtures ---

@pytest.fixture
def mock_structure(tmp_path):
    def make(index):
        return Structure(
            identity=f"round_0_struct_{index}",
            parents=[],
            parent_weights=[],
            structure_path=tmp_path / f"fake_{index}.pdb",
            sequence="ACDE",
            prob_dist=np.random.rand(4, 21),
        )
    return make


@pytest.fixture
def fape_mock():
    # Return fixed mock FAPE values (sum of indices for reproducibility)
    def fape_fn(traj_a, traj_b, aln_a, aln_b):
        return (traj_a.fake_idx + traj_b.fake_idx)
    return fape_fn


@pytest.fixture
def md_load_patch():
    with patch("your_package.fape_util.md.load") as mdload:
        def make_fake_traj(index):
            traj = MagicMock()
            traj.fake_idx = index
            return traj
        mdload.side_effect = make_fake_traj
        yield mdload


# --- Tests for compute_pairwise_fape ---

def test_compute_pairwise_fape_basic(mock_structure, fape_mock, md_load_patch):
    structs = [mock_structure(i) for i in range(3)]
    for i, s in enumerate(structs):
        s.aligned_indices = np.arange(len(s.sequence))

    fape_matrix = compute_pairwise_fape(structs, fape_mock)

    # FAPE value at (i, j) is average of (i+j) + (j+i) = i+j
    expected = np.array([
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0]
    ], dtype=float)

    np.testing.assert_array_equal(fape_matrix, expected)


def test_compute_pairwise_fape_incremental(mock_structure, fape_mock, md_load_patch):
    s0 = [mock_structure(i) for i in range(2)]
    for s in s0:
        s.aligned_indices = np.arange(len(s.sequence))

    fape_0 = compute_pairwise_fape(s0, fape_mock)

    # Add a third structure
    s1 = s0 + [mock_structure(2)]
    s1[2].aligned_indices = np.arange(len(s1[2].sequence))

    fape_1 = compute_pairwise_fape(s1, fape_mock, prev_fape_matrix=fape_0)

    assert fape_1.shape == (3, 3)
    assert fape_1[0, 1] == 1  # preserved from previous matrix
    assert fape_1[0, 2] == 2  # newly computed
    assert fape_1[2, 0] == 2


# --- Tests for shortest_fape_path_mst ---

def test_shortest_fape_path_mst_simple():
    fape_matrix = np.array([
        [0, 1, 5, 9],
        [1, 0, 2, 8],
        [5, 2, 0, 3],
        [9, 8, 3, 0],
    ], dtype=float)

    path, gap = shortest_fape_path_mst(fape_matrix, source_idx=0, target_idx=3)

    assert path == [0, 1, 2, 3]
    assert gap == (2, 3)  # largest FAPE in path: 3


def test_shortest_fape_path_mst_disconnected():
    fape_matrix = np.full((3, 3), np.inf)
    np.fill_diagonal(fape_matrix, 0)
    with pytest.raises(ValueError, match="No path from"):
        shortest_fape_path_mst(fape_matrix, 0, 2)
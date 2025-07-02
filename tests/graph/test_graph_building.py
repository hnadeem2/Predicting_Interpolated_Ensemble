import numpy as np
import pytest
from pie.graph.graph_building import compute_pairwise_fape, shortest_fape_path_mst
from pie.data_structs import Structure
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def dummy_structure(tmp_path):
    """
    Creates a dummy Structure object with aligned indices and mock PDB path.
    """
    path = tmp_path / "dummy.pdb"
    path.write_text("REMARK dummy structure")  # content doesn't matter for mocking

    prob = np.random.rand(10, 21)
    prob /= prob.sum(axis=1, keepdims=True)

    return Structure(
        identity="dummy",
        structure_path=path,
        sequence="A" * 10,
        prob_dist=prob,
        aligned_indices=np.arange(10)
    )


@patch("pie.graph.graph_building.md.load")
def test_compute_pairwise_fape_with_mocked_mdload(mock_mdload, dummy_structure):
    """
    Test compute_pairwise_fape using mocked md.load and fape_fn.
    """
    mock_traj = MagicMock()
    mock_mdload.return_value = mock_traj

    def dummy_fape_fn(traj1, traj2, idx1, idx2):
        return 1.23

    structs = [dummy_structure, dummy_structure]
    fape_matrix = compute_pairwise_fape(structs, dummy_fape_fn)

    assert fape_matrix.shape == (2, 2)
    assert fape_matrix[0, 1] == pytest.approx(1.23)
    assert fape_matrix[1, 0] == pytest.approx(1.23)
    assert fape_matrix[0, 0] == 0.0
    assert fape_matrix[1, 1] == 0.0


def test_shortest_fape_path_mst_simple_case():
    """
    Test shortest path and largest gap on a known small FAPE matrix.
    """
    matrix = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 0.5],
        [2.0, 0.5, 0.0]
    ])

    path, gap = shortest_fape_path_mst(matrix, 0, 2)

    assert path == [0, 1, 2]
    assert gap == (0, 1) or gap == (1, 2)  # both have non-zero weights, one will be the largest


def test_shortest_fape_path_mst_disconnected_graph_raises():
    """
    If MST is disconnected (e.g., diagonal-only), expect ValueError.
    """
    fape_matrix = np.array([
        [0.0, 0.0],
        [0.0, 0.0]
    ])
    with pytest.raises(ValueError):
        shortest_fape_path_mst(fape_matrix, 0, 1)

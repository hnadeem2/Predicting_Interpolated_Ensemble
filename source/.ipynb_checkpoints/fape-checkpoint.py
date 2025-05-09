import numpy as np

def extract_backbone_coordinates(traj):
    """
    Extracts the backbone atom coordinates (N, CA, C) from the first frame of a trajectory.

    Args:
        traj: An MDTraj trajectory object.

    Returns:
        np.ndarray of shape (n_residues, 3, 3): 
        Array of backbone coordinates per residue. 
        Axis 0 = residues, axis 1 = atoms (N, CA, C), axis 2 = xyz.
    """
    backbone_atoms = ['N', 'CA', 'C']
    atom_coors = np.zeros((traj.n_residues, 3, 3))
    for i, a in enumerate(backbone_atoms):
        coors = traj.xyz[0, traj.top.select(f"name {a}"), :]
        assert coors.shape[0] == traj.n_residues, f"Expected {traj.n_residues}, got {coors.shape[0]}"
        atom_coors[:, i, :] = coors

    return atom_coors


def check_rotation(rot_mat):
    """
    Verifies that a batch of matrices are valid rotation matrices.

    Args:
        rot_mat: np.ndarray of shape (N, 3, 3). Rotation matrices.

    Raises:
        AssertionError: If any matrix is not orthogonal or does not have determinant ≈ 1.
    """
    identity = np.eye(3)
    product = np.matmul(rot_mat.transpose(0, 2, 1), rot_mat)
    determinants = np.linalg.det(rot_mat)

    assert np.allclose(product, identity), "Rotation matrices are not orthogonal"
    assert np.allclose(determinants, 1.0), "Determinant may be inverted"


def ref_frames(backbone_coords):
    """
    Constructs local reference frames per residue from backbone coordinates.

    Args:
        backbone_coords: np.ndarray of shape (n_residues, 3, 3).
            Backbone atom coordinates per residue (N, CA, C).

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - rot_mat: Rotation matrices of shape (n_residues, 3, 3).
            - t: Translation vectors (CA positions), shape (n_residues, 3).
    """
    v1 = backbone_coords[:, 2] - backbone_coords[:, 1]  # C - CA
    v2 = backbone_coords[:, 0] - backbone_coords[:, 1]  # N - CA

    v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
    v2 -= v1 * np.sum(v1 * v2, axis=1, keepdims=True)
    v2 /= np.linalg.norm(v2, axis=1, keepdims=True)

    v3 = np.cross(v1, v2)

    assert v3.shape == (len(v1), 3)
    rot_mat = np.stack([v1, v2, v3], axis=-1)
    assert rot_mat.shape == (len(backbone_coords), 3, 3)

    check_rotation(rot_mat)

    t = backbone_coords[:, 1]
    return rot_mat, t


def apply_rotation(coords, R, t):
    """
    Applies rotation and translation to coordinates using a batch of local frames.

    Args:
        coords: np.ndarray of shape (n, 3, m). Coordinates to transform.
        R: np.ndarray of shape (n, 3, 3). Rotation matrices.
        t: np.ndarray of shape (n, 3). Translation vectors.

    Returns:
        np.ndarray of shape (n, 3, m): Transformed coordinates.
    """
    rotated = np.matmul(R, coords)
    translated = rotated - t[:, :, None]
    return translated


def fape(test_traj, ref_traj):
    """
    Computes the Frame Aligned Point Error (FAPE) between two trajectories.

    Args:
        test_traj: The predicted MDTraj trajectory.
        ref_traj: The ground truth/reference MDTraj trajectory.

    Returns:
        float: Mean FAPE score across all residue pairs.
    """
    test_coords = extract_backbone_coordinates(test_traj)
    ref_coords = extract_backbone_coordinates(ref_traj)
    test_R, test_t = ref_frames(test_coords)
    ref_R, ref_t = ref_frames(ref_coords)

    N = test_traj.n_residues

    # Tile to shape (N, 3, 3N): for each residue, all residue coordinates
    tiled_test_coords = np.broadcast_to(test_coords[None, :, :, :], (N, N, 3, 3)).copy().reshape((N, 3*N, 3)).transpose(0, 2, 1)
    tiled_ref_coords = np.broadcast_to(ref_coords[None, :, :, :], (N, N, 3, 3)).copy().reshape((N, 3*N, 3)).transpose(0, 2, 1)

    tiled_test_coords = apply_rotation(tiled_test_coords, test_R, test_t)
    tiled_ref_coords = apply_rotation(tiled_ref_coords, ref_R, ref_t)

    return np.mean(np.linalg.norm(tiled_test_coords - tiled_ref_coords, axis=1))

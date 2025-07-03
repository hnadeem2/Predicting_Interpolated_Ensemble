import numpy as np

def extract_backbone_coordinates(traj, chain_id="A"):
    """
    Extracts the backbone atom coordinates (N, CA, C) from the first frame of a trajectory.

    Args:
        traj: An MDTraj trajectory object.
        chain_id: str. The chain identifier.

    Returns:
        np.ndarray of shape (n_residues, 3, 3): 
        Array of backbone coordinates per residue. 
        Axis 0 = residues, axis 1 = atoms (N, CA, C), axis 2 = xyz.
    """
    chain_idx = ord(chain_id) - ord('A')
    atom_indices = traj.top.select(f"chainid == {chain_idx}")
    traj_chain = traj.atom_slice(atom_indices)
    backbone_atoms = ['N', 'CA', 'C']
    atom_coors = np.zeros((traj_chain.n_residues, 3, 3))
    for i, a in enumerate(backbone_atoms):
        coors = traj_chain.xyz[0, traj_chain.top.select(f"name {a}"), :]
        assert coors.shape[0] == traj_chain.n_residues, f"Expected {traj_chain.n_residues}, got {coors.shape[0]}"
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


def fape(test_traj, ref_traj, test_aln_idx, ref_aln_idx, chain_ids):
    """
    Computes Frame Aligned Point Error (FAPE) for aligned residues in two trajectories.

    Args:
        test_traj: MDTraj object for predicted structure.
        ref_traj: MDTraj object for reference structure.
        test_aln_idx: np.ndarray of shape (M,), indices into test_traj.n_residues.
        ref_aln_idx: np.ndarray of shape (M,), indices into ref_traj.n_residues.
        chain_ids: List[str], list of chain identifiers.

    Returns:
        float: Mean FAPE score across aligned residue pairs.
    """
    test_coords = extract_backbone_coordinates(test_traj, chain_id=chain_ids[0])  # (N1, 3, 3)
    ref_coords  = extract_backbone_coordinates(ref_traj, chain_id=chain_ids[1])   # (N2, 3, 3)

    # Subset to aligned residues
    test_coords = test_coords[test_aln_idx]  # (M, 3, 3)
    ref_coords  = ref_coords[ref_aln_idx]    # (M, 3, 3)

    # Build reference frames for aligned residues
    test_R, test_t = ref_frames(test_coords)
    ref_R, ref_t   = ref_frames(ref_coords)

    M = len(test_aln_idx)  # Number of aligned residues

    # Repeat each structure's aligned backbone for every frame
    # Shape: (M, 3, 3M)
    tiled_test_coords = np.broadcast_to(test_coords[None, :, :, :], (M, M, 3, 3)).copy().reshape((M, 3*M, 3)).transpose(0, 2, 1)
    tiled_ref_coords  = np.broadcast_to(ref_coords[None, :, :, :],  (M, M, 3, 3)).copy().reshape((M, 3*M, 3)).transpose(0, 2, 1)

    # Transform coordinates to local frames
    tiled_test_coords = apply_rotation(tiled_test_coords, test_R, test_t)
    tiled_ref_coords  = apply_rotation(tiled_ref_coords, ref_R, ref_t)

    # Compute FAPE
    return np.mean(np.linalg.norm(tiled_test_coords - tiled_ref_coords, axis=1))


def fape_from_alignment_maps(test_traj, ref_traj, test_map, ref_map, chain_ids=["A", "A"]):
    """
    Computes FAPE between two trajectories using alignment maps to a common reference.

    Args:
        test_traj: MDTraj object for predicted structure.
        ref_traj: MDTraj object for reference structure.
        test_map: List[Optional[int]] mapping ref_seq positions to test_traj residue indices.
        ref_map: List[Optional[int]] mapping ref_seq positions to ref_traj residue indices.

    Returns:
        float: Mean FAPE over aligned, ungapped positions.
    """
    if len(test_map) != len(ref_map):
        raise ValueError("Alignment maps must be the same length")

    # Extract aligned indices (positions where both maps are not None)
    aligned_positions = [
        (t_idx, r_idx) for t_idx, r_idx in zip(test_map, ref_map)
        if t_idx is not None and r_idx is not None
    ]

    if not aligned_positions:
        raise ValueError("No aligned positions found (non-None in both maps)")

    test_indices, ref_indices = zip(*aligned_positions)
    test_indices = np.array(test_indices)
    ref_indices = np.array(ref_indices)

    return fape(test_traj, ref_traj, test_indices, ref_indices, chain_ids)
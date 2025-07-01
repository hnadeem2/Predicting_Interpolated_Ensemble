import numpy as np
import mdtraj as md 
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from typing import List, Optional
from pie.data_structs import Structure


def compute_pairwise_fape(
    structures: List[Structure],
    fape_fn: callable,
    prev_fape_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Incrementally compute pairwise FAPE scores between Structure objects.

    Parameters:
        structures (List[Structure]): All structures seen so far.
        fape_fn (callable): Function computing FAPE between two MDTraj trajs.
        prev_fape_matrix (np.ndarray, optional): Previously computed symmetric FAPE matrix.

    Returns:
        np.ndarray: Updated symmetric FAPE matrix of shape (N, N).
    """
    N = len(structures)

    # Create or expand the FAPE matrix
    if prev_fape_matrix is None:
        fape_matrix = np.zeros((N, N))
    else:
        fape_matrix = np.zeros((N, N))
        old_N = prev_fape_matrix.shape[0]
        fape_matrix[:old_N, :old_N] = prev_fape_matrix

    # Compute only the missing upper triangle (i < j and fape_matrix[i, j] == 0)
    for i in range(N):
        for j in range(i + 1, N):
            if fape_matrix[i, j] == 0 and i != j:
                struct_a = structures[i]
                struct_b = structures[j]
                struct_a_traj = md.load(struct_a.structure_path)
                struct_b_traj = md.load(struct_b.structure_path)
                struct_a_aln_map = struct_a.aligned_indices
                struct_b_aln_map = struct_b.aligned_indices
                score_1 = fape_fn(struct_a_traj, struct_b_traj, struct_a_aln_map, struct_b_aln_map)
                score_2 = fape_fn(struct_b_traj, struct_a_traj, struct_b_aln_map, struct_a_aln_map)
                fape_matrix[i, j] = (score_1 + score_2) / 2
                fape_matrix[j, i] = fape_matrix[i, j]

    return fape_matrix


def shortest_fape_path_mst(fape_matrix, source_idx, target_idx):
    """
    Compute shortest path and largest gap in FAPE distance graph restricted to MST.

    Parameters:
        fape_matrix (np.ndarray): Pairwise FAPE score matrix.
        source_idx (int): Index of source structure.
        target_idx (int): Index of target structure.

    Returns:
        path (list of int): Indices of structures in shortest MST path.
        gap_indices (tuple of int): Pair of indices with largest FAPE in path.
    """
    # Step 1: Compute MST
    dist_matrix = fape_matrix.copy()
    np.fill_diagonal(dist_matrix, 0)
    mst = minimum_spanning_tree(dist_matrix).toarray()
    mst = np.maximum(mst, mst.T)  # Make symmetric for undirected path

    # Step 2: Compute shortest path on MST graph
    _, predecessors = shortest_path(mst, directed=False, return_predecessors=True)

    # Step 3: Reconstruct path
    path = []
    current = target_idx
    while current != source_idx:
        path.append(current)
        current = predecessors[source_idx, current]
        if current == -9999:
            raise ValueError(f"No path from {source_idx} to {target_idx} in MST")
    path.append(source_idx)
    path = path[::-1]

    # Step 4: Find largest gap (based on original FAPE matrix)
    max_gap = -np.inf
    gap_pair = (None, None)
    for a, b in zip(path[:-1], path[1:]):
        gap = fape_matrix[a, b]
        if gap > max_gap:
            max_gap = gap
            gap_pair = (a, b)

    return path, gap_pair
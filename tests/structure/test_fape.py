import pytest
import numpy as np
import mdtraj as md
from pie.structure.fape import extract_backbone_coordinates, ref_frames, check_rotation, apply_rotation, fape, fape_from_alignment_maps


def make_dummy_traj(n_residues=5):
    # Create dummy atoms: for each residue, 3 backbone atoms
    atom_names = ['N', 'CA', 'C']
    atoms = []
    residues = []
    chains = []
    
    coords = np.zeros((n_residues, 3, 3))

    for i in range(n_residues):
        N  = np.array([i*3.0, 0.0, 0.0])
        CA = np.array([i*3.0 + 1.0, 1.0, 0.0])
        C  = np.array([i*3.0 + 2.0, 0.5, 1.0])
        coords[i, 0, :] = N
        coords[i, 1, :] = CA
        coords[i, 2, :] = C

    topology = md.Topology()
    chain = topology.add_chain()
    for i in range(n_residues):
        residue = topology.add_residue("ALA", chain)
        for name in atom_names:
            topology.add_atom(name, md.element.carbon, residue)

    xyz = np.array(coords).reshape(1, n_residues * 3, 3)
    traj = md.Trajectory(xyz, topology)
    traj.n_residues = n_residues  # Patch attribute since we use it
    return traj
        

def test_extract_backbone_coordinates():
    traj = make_dummy_traj(4)
    coords = extract_backbone_coordinates(traj)
    assert coords.shape == (4, 3, 3)


def test_ref_frames_validity():
    backbone_coords = np.random.randn(5, 3, 3)
    rot, t = ref_frames(backbone_coords)
    assert rot.shape == (5, 3, 3)
    assert t.shape == (5, 3)
    check_rotation(rot)  # should not raise


def test_apply_rotation_identity():
    coords = np.random.rand(3, 3, 2)  # (n, 3, m)
    R = np.tile(np.eye(3)[None, :, :], (3, 1, 1))  # Identity rotation
    t = np.zeros((3, 3))  # No translation

    out = apply_rotation(coords, R, t)
    np.testing.assert_allclose(out, coords)


def test_fape_sanity():
    traj1 = make_dummy_traj(4)
    traj2 = make_dummy_traj(4)

    aligned = np.array([0, 1, 2, 3])
    score = fape(traj1, traj2, aligned, aligned)
    assert isinstance(score, float)
    assert np.allclose([score], [0.0])


def test_fape_from_alignment_maps():
    traj1 = make_dummy_traj(4)
    traj2 = make_dummy_traj(4)

    test_map = [0, 1, None, 3]
    ref_map  = [0, 1, None, 2]

    score = fape_from_alignment_maps(traj1, traj2, test_map, ref_map)
    assert isinstance(score, float)
    assert np.allclose([score], [1.60549886]) # atol=1e-8, rtol=1e-05
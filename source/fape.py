import numpy as np
import mdtraj as md
from scipy.spatial.transform import Rotation


class ReferenceFrame():
	
	
	def __init__(self, R=None, t=None):
		"""Define reference frames with rotations and translations.

		Args:
			R (scipy.spatial.transform.Rotation): rotation object.
			t (np.ndarray): array of shape (3,). Translation vector (origin of reference).

		The class does not need arguments to be initialized. Arguments can be set from three 3D arbitrary vectors
		with ReferenceFrame().from_points(v1, v2, v3) where v2 will be used as origin.
		"""
		self.R = R
		self.t = t 


	def from_points(self, v1, v2, v3):
		"""Set frame of reference from three 3D vectors.

		Args:
			v1, v2, v3 (np.ndarray): arrays of shape (3,).
		"""
		mat = np.column_stack([v1, v2, v3])
		rot_mat, _ = np.linalg.qr(mat)
		self.R = Rotation.from_matrix(rot_matrix)
		self.t = v2


	def shift_frame(self, arr):
		"""Shift a matrix of shape (N, 3) to the frame of reference of the object.

		Args:
			arr (np.ndarray): matrix of shape (N, 3) containing 3D coordinates.

		Returns:
			shifted_arr (np.ndarray): shifted coordinates.
		"""
		if self.t is None or self.R is None:
			raise ValueError("Frame of reference has not been defined.")
		
		shifted_arr = arr + self.t
		return self.R.apply(shifted_arr)


def extract_backbone_coordinates(traj):
	backbone_atoms = ['N', 'CA', 'C']
	atom_coors = np.zeros((traj.n_residues, 3, 3))
	for i, a in enumerate(backbone_atoms):
		coors = traj.xyz[0, traj.top.select(f"name {a}"), :]
		atom_coors[:, i, :] = coors

	return atom_coors


def check_rotation(rot_mat):
	identity = np.eye(3)
	product = np.matmul(rot_mat.transpose(0,2 ,1), rot_mat)
	determinants = np.linalg.determinants(rot_mat)

	assert np.allclose(product, identity), "Rotation matrices are not orthogonal"
	assert np.allclose(determinants, 1.0), "Determinant may be inverted"


def ref_frames(backbone_coords):
	v1 = backbone_coords[:, 2] - backbone_coords[:, 1]
	v2 = backbone_coords[:, 0] - backbone_coords[:, 1]
	v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
	v2 -= v1*np.dot(v1, v2, axis=1)
	v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
	assert v3.shape == (len(v1), 3)
	rot_mat = np.stack([v1, v2, v3], axis=-1)
	assert rot_mat.shape == (len(backbone_coords), 3, 3)
	check_rotation(rot_mat)
	
	t = backbone_coords[:, 1]
	return rot_mat, t


def apply_rotation(coords, R, t):
	return R @ coords + t


def fape_squared(test_traj, ref_traj):
	test_coords = extract_backbone_coordinates(test_traj)
	ref_traj = extract_backbone_coordinates(ref_traj)

	
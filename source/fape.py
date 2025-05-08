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





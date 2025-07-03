from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from typing import List

@dataclass
class Structure:
    identity: str                                   # Identifier of the structure (either template name or round_{#}_struct_{#})
    structure_path: Path                            # Path to PDB file
    sequence: str                                   # Amino acid sequence of the PDB structure
    prob_dist: np.ndarray                           # Shape (L, 21): residue identity probability distribution at each site
    aligned_indices: np.ndarray = field(init=False) # Maps sequence to reference indices
    parents: List["Structure"] = field(default=None)  # Structures mixed to produce this structure
    parent_weights: List[float] = field(init=None)  # Weights used to mix parents' probabilities



    def __post_init__(self):
        if not isinstance(self.prob_dist, np.ndarray):
            raise TypeError("prob_dist must be a numpy array")

        L = len(self.sequence)
        if self.prob_dist.shape != (L, 21):
            raise ValueError(f"prob_dist must have shape ({L}, 21), got {self.prob_dist.shape}")

        # Default aligned_indices: identity mapping
        self.aligned_indices = np.arange(L)
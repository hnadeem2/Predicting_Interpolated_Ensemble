from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from typing import Literal, List, Set, Tuple, Optional

@dataclass
class Structure:
    identity: str                                   # Identifier of the structure (either template name or round_{#}_struct_{#})
    structure_path: Path                            # Path to PDB file
    sequence: str                                   # Amino acid sequence of the PDB structure
    prob_dist: np.ndarray                           # Shape (L, 21): residue identity probability distribution at each site
    chain_id: str = "A"
    aligned_indices: np.ndarray = field(init=False) # Maps sequence to reference indices
    parents: Optional[List["Structure"]] = None  # Structures mixed to produce this structure
    parent_weights: Optional[List[float]] = None  # Weights used to mix parents' probabilities



    def __post_init__(self):
        if not isinstance(self.prob_dist, np.ndarray):
            raise TypeError("prob_dist must be a numpy array")

        L = len(self.sequence.replace("-", ""))
        if self.prob_dist.shape != (L, 21):
            raise ValueError(f"prob_dist must have shape ({L}, 21), got {self.prob_dist.shape}")

        # Default aligned_indices: identity mapping
        self.aligned_indices = np.arange(L)


@dataclass
class Round:
    round_num: int
    direction: Literal["A", "B"] # A if template 1 is an anchor, B if template 2 is an anchor
    parent_1: Structure
    parent_2: Structure
    sequences: Optional[List[str]] = None # Should be presorted
    edit_distances: Optional[List[int]] = None
    weights: Optional[List[float]] = None # Lambda values
    generated_structures: Optional[List[Structure]] = None


@dataclass
class Global:
    rounds: Optional[List[Tuple[Round]]] = None
    sequence_buffer: Optional[Dict[str, Structure]] = None

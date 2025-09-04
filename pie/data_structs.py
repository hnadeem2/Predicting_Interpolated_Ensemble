from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from typing import Literal, List, Set, Tuple, Dict, Optional

@dataclass
class Structure:
    identity: str                                   # Identifier of the structure (either template name or round_{#}{direction}_struct_{#})
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

        L = len(self.sequence)
        N = self.prob_dist.shape[0]

        if self.prob_dist.shape[1] != 21:
            raise ValueError("prob_dist must have 21 columns (AA + gap/unknown)")

        # Case 1: prob_dist already matches full sequence length
        if N == L:
            pass  # nothing to do

        # Case 2: prob_dist matches gapless sequence -> expand with "X" at gaps
        elif N == len(self.sequence.replace("-", "")):
            new_prob_dist = np.zeros((L, 21))
            j = 0
            for i, aa in enumerate(self.sequence):
                if aa == "-":
                    new_prob_dist[i, 20] = 1.0  # 'X' = unknown
                else:
                    new_prob_dist[i] = self.prob_dist[j]
                    j += 1
            self.prob_dist = new_prob_dist

        else:
            raise ValueError(
                f"prob_dist length {N} does not match sequence length {L} "
                f"or gapless length {len(self.sequence.replace('-', ''))}"
            )

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
class GlobalTracker:
    rounds: List[Tuple[Round]] = field(default_factory=list)
    sequence_buffer: Dict[str, Structure] = field(default_factory=dict)

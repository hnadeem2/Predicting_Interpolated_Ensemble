import argparse
from pathlib import Path
from typing import List
import numpy as np
import biotite.structure.io as bsio
import biotite.structure as struc
from data_structs import Structure
from prob_mixture import compute_alignment_indices, run_pmpnn, combine_features_from_indices, get_max_likelihood_seq
from predict_struct import run_boltz
from constants import PMPNN_ALPHABET

def getargs():
    parser = argparse.ArgumentParser(description="Interpolate between structural templates.")

    # Required arguments
    parser.add_argument(
        "ref_seq",
        type=str,
        help="Reference or canonical sequence to model."
    )
    parser.add_argument(
        "template_dir",
        type=Path,
        help="Path to structural templates (use 2 PDBs) for interpolation."
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)."
    )
    parser.add_argument(
        "--pmpnn_path",
        type=Path,
        default=Path("/opt/ProteinMPNN"),
        help="Path to protein_mpnn_run.py (default: /opt/ProteinMPNN)."
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of interpolation rounds (default: 10)."
    )
    parser.add_argument(
        "--interpolation_steps",
        type=int,
        default=10,
        help="Number of interpolated predictions per round (default: 10)."
    )
    parser.add_argument(
        "--boltz_script",
        type=Path,
        default=Path("./boltz.sh"),
        help="Path to boltz-2 executable script (default: ./boltz.sh)."
    )
    parser.add_argument(
        "--pmpnn_script",
        type=Path,
        default=Path("./pmpnn.sh"),
        help="Path to protein mpnn executable script (default: ./pmpnn.sh)."
    )

    return parser.parse_args()


def load_templates(template_dir: Path, ref_seq: str, **pmpnn_kwargs) -> List[Structure]:
    # Step 1: Find .pdb files in the template_dir
    pdb_files = list(template_dir.glob("*.pdb"))
    if len(pdb_files) != 2:
        raise ValueError(f"Expected exactly 2 PDB files in {template_dir}, found {len(pdb_files)}")

    structures: List[Structure] = []
    sequences: List[str] = []

    # Step 2: Loop through structure files
    for pdb_path in pdb_files:
        # Load structure using biotite
        file = bsio.load_structure(pdb_path)
        # Extract sequence with gaps
        seq = struc.get_residue_sequence(file, include_gaps=True, gap_char='-')
        seq_str = ''.join(seq)
        sequences.append(seq_str)

    # Step 3: Compute alignment indices for each structure to the reference
    alignment_indices = compute_alignment_indices(sequences, ref_seq)

    # Step 4: Run ProteinMPNN and create Structure objects
    for pdb_path, seq_str, aligned_idx in zip(pdb_files, sequences, alignment_indices):
        _, npz_path = run_pmpnn(pdb_path, **pmpnn_kwargs)
        npz_data = np.load(npz_path)
        prob_dist = np.squeeze(npz_data["probs"])
        
        if prob_dist.shape[0] != len(seq_str):
            raise ValueError(f"Shape mismatch: prob_dist has shape {prob_dist.shape}, sequence length is {len(seq_str)}")

        structure = Structure(
            identity=pdb_path.stem,
            structure_path=pdb_path,
            sequence=seq_str,
            prob_dist=prob_dist
        )
        # Manually override aligned_indices since we're supplying it explicitly
        structure.aligned_indices = aligned_idx

        structures.append(structure)

    return structures


def run_round_master(num_round, structures, args):
    
    # Find anchors
    struct_anchors = find_anchors(structures)

    # Compute mixtures
    prob_arrays = [sa.prob_dist for sa in struct_anchors]
    index_maps = [sa.aligned_indices for sa in struct_anchors]
    weights = np.linspace(0, 1, args.interpolation_steps)
    mixed_probs = [combine_features_from_indices(prob_arrays, index_maps, weights=[w, 1-w]) for w in weights]

    # Find max likelihood sequences
    max_like_seqs = [get_max_likelihood_seq(mp, PMPNN_ALPHABET) for mp in mixed_probs]

    # Save these sequences as FASTA files and run Boltz
    fasta_paths = save_boltz_input(max_like_seqs, args, num_round, struct_anchors) 

    boltz_kwargs = {
        # Fill up later
    }

    pdb_paths = [run_boltz(fp, **boltz_kwargs) for fp in fasta_paths]

    # Run ProteinMPNN
    pmpnn_kwargs = {
        # Fill up later
    }

    prob_dists = []
    for pdb_path in pdb_paths:
        _, npz_path = run_pmpnn(pdb_path, **pmpnn_kwargs)
        npz_data = np.load(npz_path)
        prob_dist = np.squeeze(npz_data["probs"])
        prob_dists.append(prob_dist)

    # Create new Structure objects
    new_structs: List[Structure] = []
    for i, (structure_path, sequence, prob_dist, w) in zip(pdb_paths, max_like_seqs, prob_dists, weights):
        identity = f"round_{num_round}_struct_{i}"
        parent_weights = [w, 1-w]

        new_struct = Structure(
            identity=identity,
            parents=struct_anchors,
            parent_weights=parent_weights,
            structure_path=structure_path,
            sequence=sequence,
            prob_dist=prob_dist,
        )

        new_structs.append(new_struct)

    structures += new_structs

    write_round_summary(structures, args)

    return structures



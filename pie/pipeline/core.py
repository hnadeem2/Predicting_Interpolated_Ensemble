from typing import List
from pathlib import Path
from biotite.structure.io import load_structure
from biotite.structure import filter_amino_acids
from biotite.sequence import ProteinSequence
import numpy as np
from pie.data_structs import Structure
from pie.sequence.predict_sequence import run_pmpnn
from pie.sequence.align import read_alignment_indices, compute_alignment_indices
from pie.sequence.sequence_utils import combine_features_from_indices, get_max_likelihood_seq
from pie.structure.predict_structure import save_boltz_input, run_boltz
from pie.graph.graph_building import compute_pairwise_fape, shortest_fape_path_mst
from pie.structure.fape import fape_from_alignment_maps as fape_fn


def load_modeled_seq(pdb_path, chain_id):
    """
    Returns a string with the amino acid sequence of a specific chain,
    including '-' characters for unmodeled residues.
    
    Assumes residue IDs are sequential integers with no insertion codes.
    
    Parameters
    ----------
    pdb_path : str or Path
        Path to the input PDB file.
    chain_id : str
        The chain ID to extract (e.g., "A").
    
    Returns
    -------
    str
        The amino acid sequence with '-' for missing residues.
    """
    # Load all atoms
    atom_array = load_structure(pdb_path)

    # Select atoms from the specified chain
    chain_atoms = atom_array[atom_array.chain_id == chain_id]
    if chain_atoms.array_length() == 0:
        raise ValueError(f"Chain '{chain_id}' not found in {pdb_path}")

    # Filter for standard amino acids
    protein_atoms = chain_atoms[filter_amino_acids(chain_atoms)]
    if protein_atoms.array_length() == 0:
        return ""

    # Get residue IDs and names
    res_ids = protein_atoms.res_id
    res_names = protein_atoms.res_name

    # Get unique residues
    unique_res_ids, indices = np.unique(res_ids, return_index=True)
    unique_res_names = res_names[indices]

    # Map res_id to 1-letter code
    res_map = {
        res_id: ProteinSequence.convert_letter_3to1(res_name)
        for res_id, res_name in zip(unique_res_ids, unique_res_names)
    }

    # Fill in missing residues with '-'
    full_range = range(min(unique_res_ids), max(unique_res_ids) + 1)
    seq = ''.join(res_map.get(i, '-') for i in full_range)

    return seq


def load_templates(template_paths: List[Path], ref_seq: str, chain_ids: List[str], aln_file: Path = None, **pmpnn_kwargs) -> List[Structure]:
    # Step 1: Check PDB files
    pdb_files = template_paths
    if len(pdb_files) != 2:
        raise ValueError(f"Expected exactly 2 PDB files in {template_dir}, found {len(pdb_files)}")

    # Check that each path exists and is a file
    for path in pdb_files:
        if not Path(path).is_file():
            raise FileNotFoundError(f"PDB file not found or is not a file: {path}")

    structures: List[Structure] = []
    sequences: List[str] = []

    # Step 2: Loop through structure files
    for pdb_path, chain_id in zip(pdb_files, chain_ids):
        # Extract sequence with gaps
        seq = load_modeled_seq(pdb_path, chain_id)
        sequences.append(seq)

    # Step 3: Compute alignment indices for each structure to the reference
    if aln_file is not None:
        alignment_indices = read_alignment_indices(aln_file, ref_seq)
    else:
        alignment_indices = compute_alignment_indices(sequences, ref_seq)

    # Step 4: Run ProteinMPNN and create Structure objects
    for pdb_path, seq_str, aligned_idx, chain_id in zip(pdb_files, sequences, alignment_indices, chain_ids):
        _, npz_path = run_pmpnn(pdb_path, pdb_path_chains=chain_id, **pmpnn_kwargs)
        npz_data = np.load(npz_path)
        prob_dist_raw = np.squeeze(npz_data["probs"])
        mpnn_mask = np.squeeze(npz_data["mask"]).astype(int)
        prob_dist = prob_dist_raw[mpnn_mask == 1]
        
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


def find_anchors(structures, cached_dist_mat=None):

    # Get FAPE matrix
    fape_matrix = compute_pairwise_fape(structures, fape_fn, cached_dist_mat)

    # Find path and gap pair
    path, gap_idx = shortest_fape_path_mst(fape_matrix, 0, 1) # 0 and 1 are the two user-provided templates

    return path, [structures[gap_idx[0]], structures[gap_idx[1]]], fape_matrix


def run_round_master(num_round, structures, cached_dist_mat, args):
    
    # Find anchors
    path, struct_anchors, fape_matrix = find_anchors(structures, cached_dist_mat)

    # Compute mixtures
    prob_arrays = [sa.prob_dist for sa in struct_anchors]
    index_maps = [sa.aligned_indices for sa in struct_anchors]
    weights = np.linspace(0, 1, args.interpolation_steps)
    mixed_probs = [combine_features_from_indices(prob_arrays, index_maps, weights=[w, 1-w]) for w in weights]

    # Find max likelihood sequences
    max_like_seqs = [get_max_likelihood_seq(mp, PMPNN_ALPHABET) for mp in mixed_probs]

    # Save these sequences as FASTA files and run Boltz
    fasta_paths = save_boltz_input(max_like_seqs, args, num_round) 
    fasta_paths_dir = os.path.dirname(fasta_paths[0])
    assert all(os.path.dirname(f) == os.path.dirname(fasta_paths_dir) for f in fasta_paths), "Not all files are in the same directory"
    
    boltz_kwargs = {
        "output_dir": Path(args.output_dir, f"round_{num_round}", "boltz", "output", f"struct_{i}"),
        "boltz_script": args.boltz_script,
        "accelerator": args.device,
    }

    pdb_paths = run_boltz(fasta_paths_dir, **boltz_kwargs)

    # Run ProteinMPNN
    
    prob_dists = []
    for pdb_path in pdb_paths:
        pmpnn_kwargs = {
            "output_dir": Path(args.output_dir, f"round_{num_round}", "pmpnn", "output", f"struct_{i}"),
            "pmpnn_path": args.pmpnn_path,
            "pmpnn_script": args.pmpnn_script,
        }

        _, npz_path = run_pmpnn(pdb_path, **pmpnn_kwargs)
        npz_data = np.load(npz_path)
        prob_dist = np.squeeze(npz_data["probs"])
        prob_dists.append(prob_dist)

    # Create new Structure objects
    new_structs: List[Structure] = []
    for i, (structure_path, sequence, prob_dist, w) in enumerate(zip(pdb_paths, max_like_seqs, prob_dists, weights)):
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

    return structures, fape_matrix, path
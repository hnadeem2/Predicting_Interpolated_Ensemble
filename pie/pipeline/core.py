from typing import List
from pathlib import Path
from biotite.structure.io import load_structure
from biotite.structure import filter_amino_acids
from biotite.sequence import ProteinSequence
import numpy as np
from pie.data_structs import Structure, Round
from pie.sequence.interpolation import find_crit_lambdas, find_interpolated_sequences, find_anchors
from pie.sequence.predict_sequence import run_pmpnn
from pie.sequence.align import read_alignment_indices, compute_alignment_indices
from pie.sequence.sequence_utils import combine_features_from_indices, get_max_likelihood_seq
from pie.structure.predict_structure import save_boltz_input, run_boltz
from pie.constants import PMPNN_ALPHABET


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
        
        if prob_dist.shape[0] != sum(i is not None for i in aligned_idx):
            raise ValueError(
                f"Shape mismatch: prob_dist has shape {prob_dist.shape}, "
                f"but {sum(i is not None for i in aligned_idx)} modeled residues were expected from alignment."
            )


        structure = Structure(
            identity=pdb_path.stem,
            structure_path=pdb_path,
            sequence=seq_str,
            prob_dist=prob_dist,
            chain_id=chain_id,
        )
        # Manually override aligned_indices since we're supplying it explicitly
        structure.aligned_indices = aligned_idx

        structures.append(structure)

    return structures


def run_round_master(num_round, global_tracker, args):
    
    # Find anchors
    struct_anchors = find_anchors(num_round, global_tracker)
    directions =  ["A", "B"] # Which user-provided template we'll use at each step

    # Run for each set of anchors
    new_rounds = []
    for anchor_set, direction in zip(struct_anchors, directions):
        # Init new round
        new_round = Round(round_num=num_round, direction=direction, parent_1=anchor_set[0], parent_2=anchor_set[1])

        # Determine sequences to fold
        crit_lambdas = find_crit_lambdas(anchor_set[0], anchor_set[1])
        pruned_seqs = find_interpolated_sequences(crit_lambdas, anchor_set[0], anchor_set[1], args.min_edit)
        # Filter sequences to avoid pre-existing ones
        final_sequences = {k: v for k, v in pruned_seqs.items() if k not in global_tracker.sequence_buffer}
        # Warn in case no new sequences are avilable and terminate early
        if not final_sequences:
            raise UserWarning(f"Round {num_round}{direction} did not yield new sequences. Moving on...")
            new_round.sequences = []
            new_round.edit_distances = []
            new_round.weights = []
            new_round.generated_structures = []
            global_tracker.rounds.append(new_round)
            continue

        # Save these sequences as FASTA files and run Boltz
        max_like_seqs = list(final_sequences.keys())
        fasta_paths = save_boltz_input(max_like_seqs, args, num_round, direction) 
        fasta_paths_dir = Path(fasta_paths[0]).parent

        boltz_kwargs = {
            "output_dir": Path(args.output_dir, f"round_{num_round}{direction}", "boltz", "output"),
            "boltz_script": args.boltz_script,
            "accelerator": args.device,
        }

        pdb_paths = run_boltz(fasta_paths_dir, **boltz_kwargs)

        # Run ProteinMPNN
    
        prob_dists = []
        for i, pdb_path in enumerate(pdb_paths):
            pmpnn_kwargs = {
                "output_dir": Path(args.output_dir, f"round_{num_round}{direction}", "pmpnn", "output", f"struct_{i}"),
                "mpnn_path": args.pmpnn_path,
                "pmpnn_script": args.pmpnn_script,
            }

            _, npz_path = run_pmpnn(pdb_path, **pmpnn_kwargs)
            npz_data = np.load(npz_path)
            prob_dist = np.squeeze(npz_data["probs"])
            prob_dists.append(prob_dist)

        # Create new Structure objects
        new_structs: List[Structure] = []
        weights = [v[0] for v in final_sequences.values()]
        edit_distances = [v[1] for v in final_sequences.values()]
        for i, (structure_path, sequence, prob_dist, w) in enumerate(zip(pdb_paths, max_like_seqs, prob_dists, weights)):
            identity = f"round_{num_round}{direction}_struct_{i}"
            parent_weights = [w, 1-w]

            new_struct = Structure(
                identity=identity,
                parents=list(anchor_set),
                parent_weights=parent_weights,
                structure_path=structure_path,
                sequence=sequence,
                prob_dist=prob_dist,
            )

            new_structs.append(new_struct)

        new_round.sequences = max_like_seqs
        new_round.edit_distances = edit_distances
        new_round.weights = weights
        new_round.generated_structures = new_structs
        new_rounds.append(new_round)

    global_tracker.rounds.extend(new_rounds)


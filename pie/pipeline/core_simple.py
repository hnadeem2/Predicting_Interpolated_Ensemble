from typing import List
from pathlib import Path
from biotite.structure.io import load_structure
from biotite.structure import filter_amino_acids
from biotite.sequence import ProteinSequence
import numpy as np
from Levenshtein import distance as edit_distance 
from Levenshtein import editops
from pie.data_structs import Structure, Round
from pie.sequence.predict_sequence import run_pmpnn
from pie.sequence.align import read_alignment_indices, compute_alignment_indices
from pie.sequence.sequence_utils import combine_features_from_indices, get_max_likelihood_seq
from pie.structure.predict_structure import save_boltz_input, run_boltz
# from pie.graph.graph_building import compute_pairwise_fape, shortest_fape_path_mst
# from pie.structure.fape import fape_from_alignment_maps as fape_fn
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


def find_crit_lambdas(template_1: Structure, template_2: Structure):
    '''Find critical values of lambda where the sequence would change sequence.

    Use formula lambda = (pij^B - pik^B) / [(pik^A - pik^B) - (pij^A - p^ij^B)] 
    for all (j,k) pairs at each position i

    Returns sorted lambda values such that lambda is in [0, 1].
    '''
    prob_1 = template_1.prob_dist
    prob_2 = template_2.prob_dist

    numerator = prob_2[:, :, None] - prob_2[:, None, :]
    diff = prob_1 - prob_2
    denominator = diff[:, :, None] - diff[:, None, :]
    l = numerator / denominator
    lambda_crit = np.sort(l[np.where(np.logical_and(l >= 0, l <= 1))])
    lambda_crit_inter = (np.asarray([0] + list(lambda_crit)[:-1]) + lambda_crit) / 2 # Use values between critical points

    return lambda_crit_inter


def compute_edit_distance(seqs, min_edit):
    """
    Given an ordered dict {sequence: value}, compute edit distance to the previous sequence.
    Returns a new dict {sequence: (lambda, edit_distance)}, 
    excluding entries with edit distance < min_edit.
    """
    new_dict = dict() # Python 3.9+ means this is ordered
    prev_seq = None
    prev_val = None

    for seq, val in seq_dict.items():
        if prev_seq is None:
            # First entry: keep with edit_distance=0
            new_dict[seq] = (val, 0)
        else:
            d = edit_distance(prev_seq, seq)
            if d >= min_edit:
                new_dict[seq] = (val, d)
        prev_seq, prev_val = seq, val

    return new_dict


def find_interpolated_sequences(lambda_crit_inter: np.ndarray, template_1: Structure, template_2: Structure, min_edit: int = 1):
    '''Use critical lambda values and probability distributions to build a nonredundant 
    set of protein sequences.

    Returns a dictionary where key is the sequence value and val is the tuple (lambda value, edit distance).
    The edit distance corresponds to the distance between two "adjacent" sequences.
    '''
    prob_1 = template_1.prob_dist
    prob_2 = template_2.prob_dist

    probs_all = (lambda_crit_inter * prob_1[:, :, None] + (1-lambda_crit_inter)*prob_2[:, :, None]).transpose(2, 0, 1)

    seqs = dict()
    alphabet = np.asarray(PMPNN_ALPHABET)
    for p, l in zip(probs_all, lambda_crit_inter):
        argmax = np.argmax(p, axis=1)
        seq = ''.join(alphabet[argmax])
        seqs[seq] = l

    pruned_seqs = compute_edit_distance(seqs, min_edit)

    return pruned_seqs


def find_anchors(num_round, global_tracker):
    '''
    Define anchors for each round. The following rules are followed:

    - Round 1: only two templates, use them as anchors.
    - Round 2: previous round only has one "direction". Repeat intermediate anchor.
    - Round n>2: define two sets of anchors.
        First set: first template provided by user + maximum edit distance step from previous round
        Second set: second template provided by user + maximum edit distance step from previous round
    '''
    round_0 = global_tracker.rounds[0][0]

    if num_round == 1:
        return [(round_0.parent_1, round_0.parent_2)]
    elif num_round == 2:
        prev_round = global_tracker.rounds[num_round-1][0]
        anchor_idx = np.argmax(prev_round.edit_distances) + 1
        common_anchor =  prev_round.generated_structures[anchor_idx]
        return [(round_0.parent_1, common_anchor), (round_0.parent_2, common_anchor)]       
    else:
        # First set of anchors
        prev_round_A = global_tracker.rounds[num_round-1][0]
        assert prev_round_A.direction == "A"
        anchor_idx = np.argmax(prev_round_A.edit_distances) + 1
        anchor_set_A = (round_0.parent_1, prev_round_A.generated_structures[anchor_idx])
        
        # Second set of anchors
        prev_round_B = global_tracker.rounds[num_round-1][1]
        assert prev_round_B.direction == "B"
        anchor_idx = np.argmax(prev_round_B.edit_distances) + 1
        anchor_set_B = (round_0.parent_2, prev_round_B.generated_structures[anchor_idx])
        return [anchor_set_A, anchor_set_B]



def run_round_master(num_round, global_tracker, args):
    
    # Find anchors
    struct_anchors = find_anchors(num_round, global_tracker)
    directions =  ["A", "B"] # Which user-provided template we'll use at each step

    # Run for each set of anchors
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
        fasta_paths = save_boltz_input(max_like_seqs, args, num_round) 
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
        global_tracker.rounds.append(new_round)
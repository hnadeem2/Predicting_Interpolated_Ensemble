# import biotite.database.rcsb as rcsb
# import biotite.structure.io.pdb as pdb
# import biotite.structure as struc
# import biotite.sequence as seq
TODO: fix imports
from typing import List
from pie.data_structs import Structure


def load_templates(template_dir: Path, ref_seq: str, aln_file: Path = None, **pmpnn_kwargs) -> List[Structure]:
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
        seq = struc.get_residue_sequence(file, include_gaps=True, gap_char='-') TODO: THIS IS WRONG!
        seq_str = ''.join(seq)
        sequences.append(seq_str)

    # Step 3: Compute alignment indices for each structure to the reference
    if aln_file is not None:
        alignment_indices = read_alignment_indices(aln_file, ref_seq)
    else:
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
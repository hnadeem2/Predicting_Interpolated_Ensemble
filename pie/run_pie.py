import argparse
from pathlib import Path
from Bio import SeqIO
from pie.data_structs import GlobalTracker, Round
from pie.pipeline.core import run_round_master, load_templates
from pie.io_utils import write_summary
from pie.structure.convert_backbone import extract_backbone_coords_to_pdb, run_cg2all
from pie.structure.minimize import minimize_all_pdbs


def getargs():
    parser = argparse.ArgumentParser(description="Interpolate between structural templates.")

    # Required arguments
    parser.add_argument(
        "ref_seq",
        type=Path,
        help="Path to reference sequence in FASTA format. The first sequence will be used."
    )
    parser.add_argument(
        "template_1",
        type=Path,
        help="Path to first structural template (PDB) for interpolation."
    )
    parser.add_argument(
        "template_2",
        type=Path,
        help="Path to second structural template (PDB) for interpolation."
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--chain_id_1",
        type=str,
        help="Chain ID from first template file.",
        default="A"
    )
    parser.add_argument(
        "--chain_id_2",
        type=str,
        help="Chain ID from second template file.",
        default="A"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)."
    )
    parser.add_argument(
        "--ligands",
        type=Path,
        default=None,
        help="Path to OPTIONAL ligand FASTA file (CCD or SMILES format). Default: None."
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
        "--min_edit_dist",
        type=int,
        default=1,
        help="Minimum edit distance between sequences considered in a single round (default: 1 = all sequences)."
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
    parser.add_argument(
        "--cg2all_script",
        type=Path,
        default=Path("./cg2all.sh"),
        help="Path to protein cg2all executable script (default: ./cg2all.sh)."
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="gpu",
        help="Device to use for computation: 'cpu' or 'gpu' (default: gpu)."
    )
    parser.add_argument(
        "--template_alignment",
        type=Path,
        default=None,
        help="Path to custom alignment for template structures (.fa/.fasta)."
    )
    parser.add_argument(
        "--msa_mode",
        type=str,
        choices=["server", "local", "empty"],
        default="server",
        help="MSA computation mode. Local is not implemented yet."
    )
    parser.add_argument(
        "--cg2all",
        action="store_true",
        help="Run CG2ALL postprocessing to pack reference sequence side chains in generated backbones. (default: False)"
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Run OpenMM to minimize structure energy. Requires --cg2all. (default: False)"
    )

    return parser.parse_args()


def read_fasta_first_seq(path: Path) -> str:
    """
    Read the first sequence from a FASTA file using Biopython.
    Returns the sequence as a plain string.
    """
    with open(path, "r") as f:
        records = list(SeqIO.parse(f, "fasta"))
    if not records:
        raise ValueError(f"No sequence found in FASTA file: {path}")
    return str(records[0].seq)


def main():
    args = getargs()

    # Init data for first round

    pmpnn_kwargs = {
            "output_dir": Path(args.output_dir, "round_0", "pmpnn", "output"),
            "mpnn_path": args.pmpnn_path,
            "pmpnn_script": args.pmpnn_script,
        }

    ref_seq = read_fasta_first_seq(args.ref_seq)

    args.ligands_str = None
    if args.ligands is not None:
        if not args.ligands.exists():
            parser.error(f"Ligands file not found: {args.ligands}")
        args.ligands_str = args.ligands.read_text(encoding="utf-8")

    structures = load_templates(
        [args.template_1, args.template_2], 
        ref_seq, 
        [args.chain_id_1, args.chain_id_2],
        **pmpnn_kwargs,
    )

    global_tracker = GlobalTracker()
    round_0 = Round(round_num=0, direction="A", parent_1=structures[0], parent_2=structures[1])
    global_tracker.rounds.append((round_0,))

    write_summary(global_tracker, args)

    # Run rounds
    for num_round in range(1, args.rounds + 1):
        print(f"Running round {num_round}")
        run_round_master(num_round, global_tracker, args)
        write_summary(global_tracker, args)

    if args.cg2all:
        # Gather all generated sequences
        gen_structures = []
        for gen_round in global_tracker.rounds:
            for gen_round_dir in gen_round:
                for gen_struct in gen_round_dir.generated_structures:
                    gen_structures.append(gen_struct)

        # Run cg2all
        output_dir = Path(args.output_dir, "cg2all")
        for structure in gen_structures:
            _ = extract_backbone_coords_to_pdb(structure, ref_seq, output_dir)
        run_cg2all(output_dir, args.cg2all_script, args.device)

    if args.cg2all and args.minimize:
        input_dir = output_dir # From cg2all
        output_dir = Path(args.output_dir, "minimized")
        minimize_all_pdbs(input_dir, output_dir)
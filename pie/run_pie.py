import argparse
from pathlib import Path
from pie.pipeline.core import run_round_master, load_templates
from pie.io_utils import write_round_summary


def getargs():
    parser = argparse.ArgumentParser(description="Interpolate between structural templates.")

    # Required arguments
    parser.add_argument(
        "ref_seq",
        type=str,
        help="Reference or canonical sequence to model."
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

    return parser.parse_args()


def main():
    args = getargs()

    pmpnn_kwargs = {
            "output_dir": Path(args.output_dir, "round_0", "pmpnn", "output"),
            "pmpnn_path": args.pmpnn_path,
            "pmpnn_script": args.pmpnn_script,
        }

    structures = load_templates(
        [args.template_1, args.template_2], 
        args.ref_seq, 
        [args.chain_id_1, args.chain_id_2]
        **pmpnn_kwargs,
        )

    write_round_summary(structures, args, 0, None)
    cached_dist_mat = None

    for num_round in range(1, args.rounds + 1):
        print(f"Running round {num_round}")
        structures, cached_dist_mat, path = run_round_master(num_round, structures, cached_dist_mat, args)
        write_round_summary(structures, args, num_round, path)
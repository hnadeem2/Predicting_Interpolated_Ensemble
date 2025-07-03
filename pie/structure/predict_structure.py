import os
import subprocess
from pathlib import Path

def save_boltz_input(seqs, args, num_round):
    """
    Save sequences to FASTA files for Boltz input.

    Parameters:
        seqs (list of str): Protein sequences.
        args: Argument object with .output_dir and .msa_mode
        num_round (int): Current round number.

    Returns:
        List[str]: Paths to the written FASTA files.
    """
    base_dir = os.path.join(args.output_dir, f"round_{num_round}", "boltz", "input")
    os.makedirs(base_dir, exist_ok=True)

    fasta_paths = []
    for i, seq in enumerate(seqs):
        fasta_path = os.path.join(base_dir, f"struct_{i}.fa")
        with open(fasta_path, "w") as f:
            if args.msa_mode == "server":
                f.write(">A|protein|\n")
            elif args.msa_mode == "empty":
                f.write(">A|protein|empty\n")
            elif args.msa_mode == "local":
                raise NotImplementError(f"{args.msa_mode} is not implemented.")
            else:
                raise ValueError(f"{args.msa_mode} is not a valid MSA mode.")
            f.write(seq + "\n")
        fasta_paths.append(fasta_path)

    return fasta_paths


def run_boltz(
    input_dir,
    output_dir,
    boltz_script,
    accelerator='gpu',
    recycling_steps=3,
    output_format='pdb',
    diffusion_samples=1,
    preprocessing_threads=1
):
    """
    Generates PDB models using Boltz for all FASTA files in a directory.

    Args:
        input_dir (str): Directory containing input FASTA files.
        output_dir (str): Directory to store Boltz output.
        boltz_script (str): Path to the Boltz shell script.
        accelerator (str, optional): Compute device, e.g., 'gpu' or 'cpu'. Defaults to 'gpu'.
        recycling_steps (int, optional): Number of recycling steps. Defaults to 3.
        output_format (str, optional): Output format of structure. Defaults to 'pdb'.
        diffusion_samples (int, optional): Number of samples to draw from diffusion. Defaults to 1.
        preprocessing_threads (int, optional): Number of CPU threads to use for preprocessing. Defaults to 1.

    Returns:
        List[str]: Paths to the generated PDB files.

    Raises:
        FileNotFoundError: If any expected PDB output is not found.
    """
    # Count input .fa or .fasta files
    fasta_files = list(Path(input_dir).glob("*.fa")) + list(Path(input_dir).glob("*.fasta"))
    if not fasta_files:
        raise FileNotFoundError(f"No FASTA files found in {input_dir}")
    
    # Run Boltz script
    subprocess.run([
        'bash',
        boltz_script,
        input_dir,
        output_dir,
        accelerator,
        str(recycling_steps),
        output_format,
        str(diffusion_samples),
        str(preprocessing_threads)
    ], check=True)
    
    # Construct expected output paths
    expected_pdbs = [
        os.path.join(
            output_dir,
            "boltz_results_input",
            "predictions",
            Path(f).stem,
            f"{Path(f).stem}_model_0.pdb"
        )
        for f in fasta_files
    ]

    missing = [p for p in expected_pdbs if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing predicted PDB files:\n" + "\n".join(missing))

    return expected_pdbs
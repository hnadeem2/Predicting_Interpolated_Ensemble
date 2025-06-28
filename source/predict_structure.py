import os
import subprocess

def save_boltz_input(
    seqs, 
    args, 
    num_round):
    """
    Save sequences to FASTA files for Boltz input.

    Parameters:
        seqs (list of str): Protein sequences.
        args: Argument object with .output_dir
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
            f.write(">A|protein|\n")
            f.write(seq + "\n")
        fasta_paths.append(fasta_path)

    return fasta_paths


def run_boltz(
    input_path,
    output_dir,
    boltz_script,
    accelerator='gpu',
    recycling_steps=3,
    output_format='pdb',
    diffusion_samples=1,
    preprocessing_threads=1):
    """
    Generates a PDB model using Boltz.

    Args:
        input_path (str): Path to the input FASTA file.
        output_dir (str): Directory to store Boltz output.
        lambda_param (float): Mixing parameter for probability interpolation.
        cache_dir (str) : Directory to store cache.
        boltz_template (str, optional): Path to the Boltz shell script template. Defaults to "boltz_template.sh".
        accelerator (str, optional): Compute device, e.g., 'gpu' or 'cpu'. Defaults to 'gpu'.
        recycling_steps (int, optional): Number of recycling steps. Defaults to 3.
        output_format (str, optional): Output format of structure. Defaults to 'pdb'.
        diffusion_samples (int, optional): Number of samples to draw from diffusion. Defaults to 1.
        preprocessing_threads (int, optional): Number of cpu threads to use for preprocessing. Defaults to 12.

    Returns:
        str: Path to the generated PDB file.

    Raises:
        FileNotFoundError: If the expected PDB output is not found.
    """
    script_path = boltz_script

    subprocess.run([
        'bash',
        script_path,
        input_path,
        output_dir,
        accelerator,
        str(recycling_steps),
        output_format,
        str(diffusion_samples),
        str(preprocessing_threads)
    ], check=True)

    fasta_base = os.path.splitext(os.path.basename(input_path))[0]
    
    pdb_path = os.path.join( Change path
        output_dir,
        f"boltz_results_{fasta_base}",
        "predictions",
        fasta_base,
        f"{fasta_base}_model_0.pdb"
    )

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Expected Boltz output {pdb_path} not found.")
    
    return pdb_path
import os
import subprocess


def run_pmpnn(
    pdb_path,
    output_dir,
    mpnn_path,
    pmpnn_script,
    seed=10, 
    temp=0.1, 
    batch_size=1
    ):
    """
    Runs ProteinMPNN on a given PDB structure.

    Args:
        pdb_path (str): Path to input PDB file.
        output_dir (str): Directory for saving outputs.
        mpnn_path (str): Path to the ProteinMPNN run script.
        pmpnn_script (str, optional): Path to the ProteinMPNN shell script.
        seed (int, optional): Random seed for reproducibility. Defaults to 10.
        temp (float, optional): Sampling temperature. Defaults to 0.1.
        batch_size (int, optional): Batch size for generation. Defaults to 1.

    Returns:
        Tuple[str, str]: Paths to the generated FASTA sequence file and NPZ probability file.
    """
    script_path = pmpnn_script
    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]

    subprocess.run([
        'bash',
        script_path,
        pdb_path,
        output_dir,
        str(temp),
        str(seed),
        str(batch_size),
        mpnn_path
    ], 
    check=True)

    fasta_file = os.path.join(f"{output_dir}", "seqs", f"{pdb_name}.fa")
    prob_file = os.path.join(f"{output_dir}", "probs", f"{pdb_name}.npz")

    if not os.path.exists(fasta_file) or not os.path.exists(prob_file):
        raise FileNotFoundError(f"Expected PMPNN outputs {fasta_file} and {prob_file} not found.")

    return fasta_file, prob_file
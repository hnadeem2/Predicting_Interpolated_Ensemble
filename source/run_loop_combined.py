from Utils import *
import numpy as np
import subprocess
import os
import sys


def run_Boltz(input_path, round_no, output_dir, lambda_param,
              boltz_template="boltz_template.sh", accelerator='gpu',
              recycling_steps=3, output_format='pdb', diffusion_samples=1):
    """
    Run the Boltz diffusion model to generate a PDB structure from a FASTA input.

    Args:
        input_path (str): Path to the input FASTA file.
        round_no (int): Interpolation round number.
        output_dir (str): Directory to store output files.
        lambda_param (float): Mixing parameter.
        boltz_template (str): Path to the Boltz shell script.
        accelerator (str): Hardware accelerator type (e.g., 'gpu').
        recycling_steps (int): Number of recycling steps in Boltz.
        output_format (str): Format of the predicted structure (default: 'pdb').
        diffusion_samples (int): Number of samples to generate.

    Returns:
        str: Path to the generated PDB file.
    """
    script_path = boltz_template
    direction = os.path.basename(output_dir).split('_')[-1]
    shared_cache = os.path.join(output_dir, f"Boltz_output_{lambda_param}", f"cache_{direction}")
    os.makedirs(shared_cache, exist_ok=True)

    subprocess.run([
        'bash', script_path, input_path, output_dir, shared_cache,
        accelerator, str(recycling_steps), output_format, str(diffusion_samples)
    ], check=True)

    fasta_base = os.path.splitext(os.path.basename(input_path))[0]
    pdb_path = os.path.join(
        output_dir,
        f"boltz_results_{fasta_base}",
        "predictions",
        fasta_base,
        f"{fasta_base}_model_0.pdb"
    )

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Expected Boltz output {pdb_path} not found.")
    
    return pdb_path


def run_ProteinMPNN(pdb_path, output_dir, pmpnn_template="pmpnn_template.sh",
                    seed=10, temp=0.1, batch_size=1, num_seq_per_target=1):
    """
    Run ProteinMPNN on a PDB structure to predict sequences and probabilities.

    Args:
        pdb_path (str): Path to the input PDB file.
        output_dir (str): Directory to store outputs.
        pmpnn_template (str): Shell script for ProteinMPNN execution.
        seed (int): Random seed for reproducibility.
        temp (float): Sampling temperature.
        batch_size (int): Number of sequences to generate per batch.

    Returns:
        tuple[str, str]: Paths to the output FASTA and probabilities (.npz).
    """
    script_path = pmpnn_template
    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]

    subprocess.run([
        'bash', script_path, pdb_path, output_dir,
        str(temp), str(seed), str(batch_size), str(num_seq_per_target)
    ], check=True)

    return (
        os.path.join(output_dir, "seqs", f"{pdb_name}.fa"),
        os.path.join(output_dir, "probs", f"{pdb_name}.npz")
    )


def mix_prob(path_to_prob1, path_to_prob2, lambda_param, round_no, output_dir, label="A",
            msa_mode="mmseqs"):
    """
    Mix two ProteinMPNN probability distributions with a weighted average.

    Args:
        path_to_prob1 (str): Path to the first probability .npz file.
        path_to_prob2 (str): Path to the second probability .npz file.
        lambda_param (float): Mixing coefficient (0–1).
        round_no (int): Round number of interpolation.
        output_dir (str): Output directory for saving the mixed FASTA.
        label (str): Direction label ("A" or "B").
        msa_mode (str): One of 'mmseqs' or 'pmpnn' (synthetic MSA).
    Returns:
        str: Path to the mixed FASTA file.
    """
    prob1 = np.squeeze(np.load(path_to_prob1)['probs'])
    prob2 = np.squeeze(np.load(path_to_prob2)['probs'])

    mixed_prob = lambda_param * prob1 + (1 - lambda_param) * prob2

    seq = sequence_list()
    ml_seq_idx = np.argmax(mixed_prob, axis=1)
    ml_seq = [''.join([seq[i] for i in ml_seq_idx])]

    fasta_dir = os.path.join(output_dir, f'fasta_Boltz_input_{lambda_param}', f'round{round_no}_{label}')
    os.makedirs(fasta_dir, exist_ok=True)
    fasta_path = os.path.join(fasta_dir, f'mixed_fasta_round{round_no}.fasta')
    msa_path = ""

    if msa_mode == 'pmpnn':
        # Build a synthetic MSA from mixed prob. output
        msa_path = os.path.join(fasta_dir, f'pmpnn_msa{round_no}.a3m')
        build_synthetic_msa(mixed_prob, msa_path)

    fasta_from_seq(ml_seq, filename=fasta_path, msa_path=msa_path)


    return fasta_path


def main(rounds, lambda_param, s1_pdb, s2_pdb, output_dir, msa_mode):
    """
    Run bidirectional interpolation between two structures using ProteinMPNN and Boltz.

    Args:
        rounds (int): Total number of interpolation rounds.
        lambda_param (float): Mixing coefficient for probability blending.
        s1_pdb (str): Path to PDB file of structure 1.
        s2_pdb (str): Path to PDB file of structure 2.
        output_dir (str): Root directory for storing all outputs.
        msa_mode (str): type of MSA to use ('mmseqs' or 'pmpnn').
    """
    boltz_template = "boltz_template.sh" if msa_mode == 'mmseqs' else "boltz_template_custom_msa.sh"


    for direction in ["A", "B"]:  # A: s1 -> new, B: s2 -> new
        print(f"\nStarting interpolation direction {direction}...")

        anchor_pdb = s1_pdb if direction == "A" else s2_pdb
        changing_pdb = s2_pdb if direction == "A" else s1_pdb

        anchor_output = os.path.join(output_dir, f'pMPNN_output_{lambda_param}', f'{direction}_anchor')
        changing_output = os.path.join(output_dir, f'pMPNN_output_{lambda_param}', f'{direction}_changing')

        _, anchor_prob = run_ProteinMPNN(anchor_pdb, anchor_output)
        _, changing_prob = run_ProteinMPNN(changing_pdb, changing_output)

        for round_num in range(1, rounds + 1):
            print(f"\n=== Round {round_num} ({direction}) ===")

            round_output_dir = os.path.join(output_dir, f'Boltz_output_{lambda_param}', f'round{round_num}_{direction}')
            fasta_path = mix_prob(anchor_prob, changing_prob, lambda_param, round_num, output_dir, direction, msa_mode)
            new_pdb_path = run_Boltz(fasta_path, round_num, round_output_dir, lambda_param, boltz_template=boltz_template)

            latest_output_dir = os.path.join(output_dir, f'pMPNN_output_{lambda_param}', f'round{round_num}_{direction}_mpnn')
            new_seq_path, latest_prob_path = run_ProteinMPNN(new_pdb_path, latest_output_dir)

            changing_prob = latest_prob_path  # Update for next round


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run bidirectional ProteinMPNN interpolation.")
    parser.add_argument("--s1_pdb", type=str, required=True, help="Path to structure 1 PDB")
    parser.add_argument("--s2_pdb", type=str, required=True, help="Path to structure 2 PDB")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--msa_mode", type=str, default="mmseqs", help="'mmseqs' or 'pmpnn'")
    # parser.add_argument("--pmpnn_run_path", type=str, required=False, default="", help="Path to protein_mpnn_run.py")
    parser.add_argument("--rounds", type=int, required=True, help="Total number of interpolation rounds")
    parser.add_argument("--lambda_param_list", type=float, nargs='+', required=True,
                        help="List of lambda mixing parameters (space-separated, e.g. 0.1 0.3 0.5)")

    args = parser.parse_args()

    for l in args.lambda_param_list:
        main(args.rounds, l, args.s1_pdb, args.s2_pdb, args.output_dir, args.msa_mode)
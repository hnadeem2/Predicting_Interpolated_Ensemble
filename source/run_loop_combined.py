from Utils import *
import numpy as np
import subprocess
import os
import sys

def run_Boltz(input_path,
              round_no,
              output_dir,
              lambda_param,
              boltz_template="boltz_template.sh",
              accelerator='gpu',
              recycling_steps=3,
              output_format='pdb',
              diffusion_samples=1):
    """
    Generates a PDB model using Boltz.

    Args:
        input_path (str): Path to the input FASTA file.
        round_no (int): Current interpolation round number.
        output_dir (str): Directory to store Boltz output.
        lambda_param (float): Mixing parameter for probability interpolation.
        boltz_template (str, optional): Path to the Boltz shell script template. Defaults to "boltz_template.sh".
        accelerator (str, optional): Compute device, e.g., 'gpu' or 'cpu'. Defaults to 'gpu'.
        recycling_steps (int, optional): Number of recycling steps. Defaults to 3.
        output_format (str, optional): Output format of structure. Defaults to 'pdb'.
        diffusion_samples (int, optional): Number of samples to draw from diffusion. Defaults to 1.

    Returns:
        str: Path to the generated PDB file.

    Raises:
        FileNotFoundError: If the expected PDB output is not found.
    """
    script_path = boltz_template

    direction = os.path.basename(output_dir).split('_')[-1]
    shared_cache = f"{output_dir}/Boltz_output_{lambda_param}/cache_{direction}"
    os.makedirs(shared_cache, exist_ok=True)

    subprocess.run([
        'bash',
        script_path,
        input_path,
        output_dir,
        shared_cache,
        accelerator,
        str(recycling_steps),
        output_format,
        str(diffusion_samples)
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


def run_ProteinMPNN(pdb_path,
                    output_dir,
                    mpnn_path,
                    pmpnn_template="pmpnn_template.sh",
                    seed=10, 
                    temp=0.1, 
                    batch_size=1):
    """
    Runs ProteinMPNN on a given PDB structure.

    Args:
        pdb_path (str): Path to input PDB file.
        output_dir (str): Directory for saving outputs.
        mpnn_path (str): Path to the ProteinMPNN run script.
        pmpnn_template (str, optional): Path to the ProteinMPNN shell script template. Defaults to "pmpnn_template.sh".
        seed (int, optional): Random seed for reproducibility. Defaults to 10.
        temp (float, optional): Sampling temperature. Defaults to 0.1.
        batch_size (int, optional): Batch size for generation. Defaults to 1.

    Returns:
        Tuple[str, str]: Paths to the generated FASTA sequence file and NPZ probability file.
    """
    script_path = pmpnn_template
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
    ], check=True)

    return f"{output_dir}/seqs/{pdb_name}.fa", f"{output_dir}/probs/{pdb_name}.npz"


def mix_prob(path_to_prob1, path_to_prob2, lambda_param, round_no, output_dir, chain_dict, label="A"):
    """
    Mixes two probability distributions using a weighted average and saves the resulting sequence as a FASTA file.

    Args:
        path_to_prob1 (str): Path to the first .npz probability file.
        path_to_prob2 (str): Path to the second .npz probability file.
        lambda_param (float): Mixing parameter (0-1).
        round_no (int): Current interpolation round number.
        output_dir (str): Directory to save output FASTA.
        chain_dict (dict): Chain information to format FASTA file.
        label (str, optional): Direction label ("A" or "B"). Defaults to "A".

    Returns:
        str: Path to the generated FASTA file.
    """
    if '4IH4' in os.path.basename(path_to_prob1):
         prob1 = fix_ATD_seq_prob(np.squeeze(np.load(path_to_prob1)['probs']))
    else:
         prob1 = np.squeeze(np.load(path_to_prob1)['probs'])
 
    if '4IH4' in os.path.basename(path_to_prob2):
         prob2 = fix_ATD_seq_prob(np.squeeze(np.load(path_to_prob2)['probs']))
    else:
         prob2 = np.squeeze(np.load(path_to_prob2)['probs'])
 
     # Mix the probabilities

    mixed_prob = lambda_param * prob1 + (1 - lambda_param) * prob2

    seq = sequence_list()
    ml_seq_idx = np.argmax(mixed_prob, axis=1)
    ml_seq = ''.join([seq[i] for i in ml_seq_idx])

    fasta_dir = f'{output_dir}/fasta_Boltz_input_{lambda_param}/round{round_no}_{label}'
    os.makedirs(fasta_dir, exist_ok=True)

    fasta_path = f'{fasta_dir}/mixed_fasta_round{round_no}.fasta'
    fasta_from_seq(sequence=ml_seq, chain_dict=chain_dict, filename=fasta_path)
    
    return fasta_path


def main(rounds, lambda_param, s1_pdb, s2_pdb, pmpnn_run_path, output_dir):
    """
    Main driver for bidirectional ProteinMPNN interpolation with Boltz structure generation.

    Args:
        rounds (int): Number of interpolation rounds to run.
        lambda_param (float): Mixing parameter (0-1).
        s1_pdb (str): Path to structure 1 PDB file.
        s2_pdb (str): Path to structure 2 PDB file.
        pmpnn_run_path (str): Path to the ProteinMPNN run script.
        output_dir (str): Directory to store all outputs.
    """
    ROUNDS = rounds 

    try:
        chain_dict = compare_pdbs(s1_pdb, s2_pdb)
    except ValueError as e:
        print(f"Error: {e}")
        #sys.exit()

    for direction in ["A", "B"]:  # A: s1 -> new, B: s2 -> new
        print(f"\nStarting interpolation direction {direction}...")

        if direction == "A":
            anchor_pdb = s1_pdb
            changing_pdb = s2_pdb
        else:
            anchor_pdb = s2_pdb
            changing_pdb = s1_pdb

        _, anchor_prob = run_ProteinMPNN(pdb_path=anchor_pdb,
                                         output_dir=f'{output_dir}/pMPNN_output_{lambda_param}/{direction}_anchor',
                                         mpnn_path=pmpnn_run_path)

        _, changing_prob = run_ProteinMPNN(pdb_path=changing_pdb,
                                           output_dir=f'{output_dir}/pMPNN_output_{lambda_param}/{direction}_changing',
                                           mpnn_path=pmpnn_run_path)

        for round_num in range(1, ROUNDS + 1):
            print(f"\n=== Round {round_num} ({direction}) ===")

            round_output_dir = f'{output_dir}/Boltz_output_{lambda_param}/round{round_num}_{direction}'

            fasta_path = mix_prob(anchor_prob, changing_prob,
                                  lambda_param=lambda_param,
                                  round_no=round_num,
                                  label=direction,
                                  output_dir=output_dir,
                                  chain_dict=chain_dict)

            new_pdb_path = run_Boltz(input_path=fasta_path,
                                     round_no=round_num,
                                     output_dir=round_output_dir,
                                     lambda_param=lambda_param)

            latest_output_dir = f'{output_dir}/pMPNN_output_{lambda_param}/round{round_num}_{direction}_mpnn'
            new_seq_path, latest_prob_path = run_ProteinMPNN(pdb_path=new_pdb_path,
                                                              output_dir=latest_output_dir,
                                                              mpnn_path=pmpnn_run_path)

            changing_prob = latest_prob_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run bidirectional ProteinMPNN interpolation.")
    parser.add_argument("--s1_pdb", type=str, required=True, help="Path to structure 1 PDB")
    parser.add_argument("--s2_pdb", type=str, required=True, help="Path to structure 2 PDB")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--pmpnn_run_path", type=str, required=True, help="Path to protein_mpnn_run.py")
    parser.add_argument("--rounds", type=int, required=True, help="Total number of interpolation rounds")
    parser.add_argument("--lambda_param_list", type=float, nargs='+', required=True,
                        help="List of lambda mixing parameters (space-separated, e.g. 0.1 0.3 0.5)")

    args = parser.parse_args()
    
    for l in args.lambda_param_list:
        main(args.rounds, l, args.s1_pdb, args.s2_pdb, args.pmpnn_run_path, args.output_dir)

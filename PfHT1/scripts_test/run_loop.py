from Utils import *
import numpy as np
import subprocess
import os
import sys

def run_Boltz(input_path,
              output_dir,
              boltz_template="boltz_template.sh",
              accelerator='gpu',
              recycling_steps=3,
              output_format='pdb',
              diffusion_samples=1):

    """
    Generate PDB model from Boltz.
    Returns: path_to_pdb_model (str)
    """

    script_path = boltz_template

    # Infer direction from output_dir (e.g., round3_A → A)
    direction = os.path.basename(output_dir).split('_')[-1]
    shared_cache = f"Boltz_output/cache_{direction}"

    # Create the cache folder if it doesn't exist
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
                    pmpnn_template="pmpnn_template.sh",
                    seed=10, 
                    temp=0.1, 
                    batch_size=1):
    """
    Runs ProteinMPNN.
    Returns: path to seq (str), path to probabilities (str)
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
        str(batch_size)
    ], check=True)

    return f"{output_dir}/seqs/{pdb_name}.fa", f"{output_dir}/probs/{pdb_name}.npz"


def mix_prob(path_to_prob1, path_to_prob2, lambda_param):
    """
    Mix two probability distributions using a weighted average.
    Applies ASN bug fix to any probability file coming from 4IH4.
    """
    prob1 = np.squeeze(np.load(path_to_prob1,allow_pickle=True)['probs'])
    prob2 = np.squeeze(np.load(path_to_prob2,allow_pickle=True)['probs'])

    # Mix the probabilities
    mixed_prob = lambda_param * prob1 + (1 - lambda_param) * prob2

    # Convert to sequence
    seq = sequence_list()
    ml_seq_idx = np.argmax(mixed_prob, axis=1)
    ml_seq = [''.join([seq[i] for i in ml_seq_idx])]

    # Save FASTA
    fasta_dir = f'fasta_Boltz_input/'
    os.makedirs(fasta_dir, exist_ok=True)

    fasta_path = f'{fasta_dir}/lambda{lambda_param}.fasta'
    fasta_from_seq(ml_seq, filename=fasta_path)

    return fasta_path

def main(lambda_param_list, s1_pdb, s2_pdb):

    _, s1_prob = run_ProteinMPNN(pdb_path=s1_pdb, output_dir=f'pMPNN_output/s1')
    _, s2_prob = run_ProteinMPNN(pdb_path=s2_pdb, output_dir=f'pMPNN_output/s2')

    for lambda_param in lambda_param_list:
        fasta_path = mix_prob(s1_prob, s2_prob, lambda_param=lambda_param)

    







if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run bidirectional ProteinMPNN interpolation.")
    parser.add_argument("--lambda_param_list", type=float, nargs='+', required=True,
                        help="List of lambda mixing parameters (space-separated, e.g. 0.1 0.3 0.5)")
    parser.add_argument("--s1_pdb", type=str, required=True, help="Path to structure 1 PDB")
    parser.add_argument("--s2_pdb", type=str, required=True, help="Path to structure 2 PDB")
    
    args = parser.parse_args()

    # Pass the whole list to main
    main(args.lambda_param_list, args.s1_pdb, args.s2_pdb)


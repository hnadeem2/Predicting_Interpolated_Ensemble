from Utils import *
import numpy as np
import subprocess
import os
import sys

def run_Boltz(input_path,
              round_no,
              output_dir,
              lambda_param,
              cache_dir,
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
    #shared_cache = f"{output_dir}/Boltz_output_{lambda_param}/cache_{direction}"
    shared_cache = cache_dir
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
                    mpnn_path,
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
        str(batch_size),
        mpnn_path
    ], check=True)

    return f"{output_dir}/seqs/{pdb_name}.fa", f"{output_dir}/probs/{pdb_name}.npz"


def mix_prob(path_to_prob1, path_to_prob2, lambda_param, round_no, output_dir, label="A"):
    """
    Mix two probability distributions using a weighted average.
    Applies ASN bug fix to any probability file coming from 4IH4.
    """
    # Load and fix if needed
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

    # Convert to sequence
    seq = sequence_list()
    ml_seq_idx = np.argmax(mixed_prob, axis=1)
    ml_seq = [''.join([seq[i] for i in ml_seq_idx])]

    # Save FASTA
    fasta_dir = f'{output_dir}/fasta_Boltz_input_{lambda_param}/round{round_no}_{label}'
    os.makedirs(fasta_dir, exist_ok=True)

    fasta_path = f'{fasta_dir}/mixed_fasta_round{round_no}.fasta'
    fasta_from_seq(ml_seq, filename=fasta_path)

    return fasta_path

def main(rounds, lambda_param, s1_pdb, s2_pdb,pmpnn_run_path,output_dir):
    
    ROUNDS = rounds 

    for direction in ["A", "B"]:  # A: s1 -> new, B: s2 -> new
        print(f"\nStarting interpolation direction {direction}...")

        if direction == "A":
            anchor_pdb = s1_pdb
            changing_pdb = s2_pdb
        else:
            anchor_pdb = s2_pdb
            changing_pdb = s1_pdb

        _, anchor_prob = run_ProteinMPNN(pdb_path=anchor_pdb, output_dir=f'{output_dir}/pMPNN_output_{lambda_param}/{direction}_anchor',mpnn_path=pmpnn_run_path)
        _, changing_prob = run_ProteinMPNN(pdb_path=changing_pdb, output_dir=f'{output_dir}/pMPNN_output_{lambda_param}/{direction}_changing',mpnn_path=pmpnn_run_path)

        for round_num in range(1, ROUNDS + 1):
            print(f"\n=== Round {round_num} ({direction}) ===")

            round_output_dir = f'{output_dir}/Boltz_output_{lambda_param}/round{round_num}_{direction}'
            cache_dir = f"{output_dir}/cache"
            fasta_path = mix_prob(anchor_prob, changing_prob, lambda_param=lambda_param, round_no=round_num, label=direction,output_dir=output_dir)
            new_pdb_path = run_Boltz(input_path=fasta_path, round_no=round_num, output_dir=round_output_dir,lambda_param=lambda_param,cache_dir=cache_dir)

            latest_output_dir = f'{output_dir}/pMPNN_output_{lambda_param}/round{round_num}_{direction}_mpnn'
            new_seq_path, latest_prob_path = run_ProteinMPNN(pdb_path=new_pdb_path, output_dir=latest_output_dir,mpnn_path=pmpnn_run_path)

            # Update for next round
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
        main(args.rounds, l, args.s1_pdb, args.s2_pdb,args.pmpnn_run_path,args.output_dir)






from Utils import*
import numpy as np
import subprocess
import os
import sys

def run_Boltz(input_path,
              round_no,
              boltz_template="boltz_template.sh",
              output_dir="boltz_output",
              accelerator='gpu',
              recycling_steps=3,
              output_format='pdb',
              diffusion_samples=1):
    """
    Generate PDB model from Boltz.
    Returns: path_to_pdb_model (str)
    """

    script_path = boltz_template
    cache = f"{output_dir}/model_cache"
    subprocess.run([
        'bash',
        script_path,
        input_path,
        output_dir,
        cache,
        accelerator,
        str(recycling_steps),
        output_format,
        str(diffusion_samples)
    ], check=True)

    return os.path.join(output_dir, f"boltz_results_mixed_fasta_round1/predictions/mixed_fasta_round{round_no}/mixed_fasta_round{round_no}_model_0.pdb")

def run_ProteinMPNN(pdb_path,
                    pmpnn_template="pmpnn_template.sh",
                    seed=10,
                    output_dir="pmpnn_output", 
                    temp=0.1, 
                    batch_size=1):

    """
    Runs ProteinMPNN
    Returns: path to seq (str), path to probabilities (str)
    """

    script_path = pmpnn_template  # Path to your bash script
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

    return os.path.join(output_dir, f"seqs/{pdb_name}.fa"),  os.path.join(output_dir, f"probs/{pdb_name}.npz")

def mix_prob(path_to_prob1, path_to_prob2, lambda_param, round_no):
    """
    Mix two probability distributions using a weighted average. Compute the resulting sequence and save as fasta input for Boltz
    new_prob = lambda_param * prob1 + (1 - lambda_param) * prob2
    Returns: new_prob (array or appropriate data structure)
    """
    prob1 = np.squeeze(np.load(path_to_prob1)['probs'])
    prob2 = np.squeeze(np.load(path_to_prob2)['probs'])
    prob2 = fix_ATD_seq_prob(prob2)  # to fix ASN issue
    
    mixed_prob = lambda_param *prob1 + (1 - lambda_param) * prob2
    seq = sequence_list()
    ml_seq_idx = np.argmax(mixed_prob,axis=1) 
    ml_seq = [seq[i] for i in ml_seq_idx]
    ml_seq = [''.join(ml_seq)]


    fasta_dir = f'fasta_Boltz_input/round{round_no}'
    if not os.path.exists(fasta_dir):
        os.makedirs(fasta_dir)
    
    fasta_path = f'{fasta_dir}/mixed_fasta_round{round_no}.fasta'
    fasta_from_seq(ml_seq,filename=fasta_path)

    return fasta_path



def interpolate(s1, s2, lambda_list, T, *args):
    """
    Main function to generate interpolated structures.
    
    Parameters:
        s1 (str): Path to PDB of state 1
        s2 (str): Path to PDB of state 2
        lambda_list (list of float): Interpolation weights
        T (float): Sampling temperature
    """
    seq1_path, prob1_path = run_PMPNN(s1, *args)
    seq2_path, prob2_path = run_PMPNN(s2, *args)

    for lambda_param in lambda_list:
        interp_prob = mix_prob(prob1_path, prob2_path, lambda_param, *args)
        interp_seq = sample_from_prob(interp_prob, T, *args)
        interp_struct_ca = run_Boltz(interp_seq, *args)
        interp_struct_aa = ca_to_aa(interp_struct_ca, *args)
        

# def main():
#     ROUND = 1
#     mpnn_output = f'pMPNN_output/round{ROUND}'
#     seq_path_1, prob_path_1 = run_ProteinMPNN(pdb_path = "../structures/5HZG.A._modified.pdb",output_dir=f'{mpnn_output}/1_output')
#     seq_path_2, prob_path_2 = run_ProteinMPNN(pdb_path = "../structures/4IH4.A.pdb",output_dir=f'{mpnn_output}/2_output')
#     fasta_path = mix_prob(prob_path_1,prob_path_2,lambda_param=1.0,round_no=ROUND)
    

#     pdb_path = run_Boltz(input_path=fasta_path)


def main():
    NUM_ROUNDS = 3
    original_struc1 = "../structures/5HZG.A._modified.pdb"
    original_struc2 = "../structures/4IH4.A.pdb"

    # --- DIRECTION 1: struc2 → struc1 ---
    print("Precomputing MPNN on struc1 (Direction 1)")
    seq_path_1_fixed, prob_path_1_fixed = run_ProteinMPNN(
        pdb_path=original_struc1,
        output_dir="pMPNN_output/dir1_fixed_struc1"
    )

    print(seq_path_1_fixed)
    print(prob_path_1_fixed)


    struc2 = original_struc2
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"[Dir1] ROUND {round_num}: struc2 → struc1")
        out_dir = f"pMPNN_output/dir1_round{round_num}"
        seq_path_2, prob_path_2 = run_ProteinMPNN(struc2, f"{out_dir}/2_output")
        
        fasta_path = mix_prob(prob_path_1_fixed, prob_path_2, lambda_param=1.0, round_no=round_num)
        struc3 = run_Boltz(fasta_path)
        struc2 = struc3  # update for next round

    # # --- DIRECTION 2: struc1 → struc2 ---
    # print("Precomputing MPNN on struc2 (Direction 2)")
    # seq_path_2_fixed, prob_path_2_fixed = run_ProteinMPNN(
    #     pdb_path=original_struc2,
    #     output_dir="pMPNN_output/dir2_fixed_struc2"
    # )

    # struc1 = original_struc1
    # for round_num in range(1, NUM_ROUNDS + 1):
    #     print(f"[Dir2] ROUND {round_num}: struc1 → struc2")
    #     out_dir = f"pMPNN_output/dir2_round{round_num}"
    #     seq_path_1, prob_path_1 = run_ProteinMPNN(struc1, f"{out_dir}/1_output")

    #     fasta_path = mix_prob(prob_path_1, prob_path_2_fixed, lambda_param=1.0, round_no=round_num)
    #     struc3 = run_Boltz(fasta_path)
    #     struc1 = struc3  # update for next round

    # print("Bidirectional evolution completed.")


if __name__ == '__main__':
    main()








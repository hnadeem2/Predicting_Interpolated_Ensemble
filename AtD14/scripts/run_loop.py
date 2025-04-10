
from Utils import*
import numpy as np
import subprocess
import os


def run_Boltz(sequence, *args):
    """
    Generate PDB model from Boltz.
    Returns: path_to_pdb_model (str)
    """
    pass


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

def mix_prob(path_to_prob1, path_to_prob2, lambda_param, *args):
    """
    Mix two probability distributions using a weighted average.
    new_prob = lambda_param * prob1 + (1 - lambda_param) * prob2
    Returns: new_prob (array or appropriate data structure)
    """
    prob1 = np.squeeze(np.load(path_to_prob1)['probs'])
    prob2 = np.squeeze(np.load(path_to_prob2)['probs'])
    prob2 = fix_ATD_seq_prob(prob2)  # to fix ASN issue
    
    mixed_prob = lambda_param *prob1 + (1 - lambda_param) * prob2
    print(mixed_prob.shape)    

        



    # seq,_ = sequence_list()
    # ml_seq_idx = np.argmax(prob1,axis=1) 
    # ml_seq = [seq[i] for i in ml_seq_idx]
    # print(ml_seq)

def ca_to_aa(ca_pdb_path, *args):
    """
    Convert Carbon Alpha (Cα) PDB structure to all-atom representation.
    Returns: path_to_all_atom_pdb (str)
    """
    pass


def sample_from_prob(prob_distribution, T, *args):
    """
    Sample a sequence from a probability distribution at temperature T.
    Returns: sampled_sequence (str)
    """
    pass


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
        

def main():
    seq_path_1, prob_path_1 = run_ProteinMPNN(pdb_path = "../structures/5HZG.A._modified.pdb",output_dir='1_output')
    seq_path_2, prob_path_2 = run_ProteinMPNN(pdb_path = "../structures/4IH4.A.pdb",output_dir='2_output')
    mix_prob(prob_path_1,prob_path_2,lambda_param=0.5)

if __name__ == '__main__':
    main()
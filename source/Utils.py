from Bio import SeqIO
import numpy as np

def parse_fasta(in_fasta, out_fasta, custom_header):
    """
    Parses output fasta from PMPNN and prepares it for Boltz input
    """
    records = list(SeqIO.parse(in_fasta, "fasta"))
    if len(records) >= 2:
        record = records[1]
        record.id = custom_header  # sets the identifier part
        record.description = custom_header  # sets the full header
        SeqIO.write(record, out_fasta, "fasta")
        print("Processed fasta file saved.")
    else:
        print(f"Less than two sequences in the file. Check {out_fasta}")


def boltz_input_from_fasta(in_fasta, out_fasta, out_a3m, custom_header):
    """
    Parses output fasta from PMPNN and prepares it for Boltz input.
    This function differs from `parse_fasta` in that the MSA (.a3m file) is constructed from 
    ProteinMPNN output directly and the fasta input for Boltz contains the path to this MSA. 
    """
    records = list(SeqIO.parse(in_fasta, "fasta"))
    if len(records) >= 2:
        record = records[1]
        record.id = custom_header  # sets the identifier part
        record.description = custom_header  # sets the full header
        SeqIO.write(record, out_fasta, "fasta")
        print("Processed fasta file saved.")
    else:
        print(f"Less than two sequences in the file. Check {out_fasta}")


def sequence_list():
    """
    returns sequence list
    """
    seq_single = list("ACDEFGHIKLMNPQRSTVWYX") # TODO: move to constants.py 

    return seq_single


def fix_ATD_seq_prob(prob):
    """
    add a row to the probability matrix to account for a missing residue
    """
    row, col = prob.shape
    arr = np.zeros((1, col))
    arr[:,11] = 1 # setting probability for 'ASN' or 'N' to 1 
    new_prob = np.vstack([arr,prob])

    return new_prob
    

def fasta_from_seq(sequence_list, filename, msa_path=""):
    """
    Creates and saves a fasta file for input to Boltz, each sequence in the sequence list is written as a chain
    """
    with open(filename, 'w') as f:
        for i, seq in enumerate(sequence_list):
            identifier = chr(65 + i)  # A, B, C, ...
            f.write(f">{identifier}|protein|{msa_path}\n{seq}\n")


def build_synthetic_msa(prob, outpath, num_samples=1000):
    """
    Constructs a synthetic MSA by sampling residues at each position
    based on the probability distribution provided in `prob`.

    Args:
        prob (np.ndarray): Array of shape (1, N, A), where N is the sequence length,
                           and A is the alphabet size. Contains probabilities of each
                           residue at each position.
        outpath (str): Path to save the generated .a3m file.
        num_samples (int): Number of synthetic sequences to generate (default: 1000).
    """
    alphabet = sequence_list()

    if prob.ndim != 2:
        raise ValueError("`prob` must be of shape (seq_len, alphabet_size)")

    N, A = prob.shape[0], prob.shape[1]
    if len(alphabet) != A:
        raise ValueError("Length of `alphabet` must match alphabet size in `prob`")

    # Check all non-zero distributions are properly normalized
    tol = 1e-6
    non_zero_mask = prob.sum(axis=1) > tol
    # assert np.allclose(prob[non_zero_mask].sum(axis=1), 1.0, atol=tol), \
    #     f"Some non-zero probability distributions do not sum to 1 within {tol} tolerance."

    msa_sequences = []
    for _ in range(num_samples):
        sampled_sequence = ''
        for pos in range(N):
            distribution = prob[pos]
            total = distribution.sum()
            if total == 0:
                sampled_sequence += 'X'
            else:
                distribution = distribution / total  # Normalize
                residue = np.random.choice(alphabet, p=distribution)
                sampled_sequence += residue
        msa_sequences.append(sampled_sequence)

    # Save to .a3m format
    with open(outpath, 'w') as f:
        for i, seq in enumerate(msa_sequences):
            f.write(f">sample_{i+1}\n{seq}\n")




# input struc1 struc2
# seq1 = mpnn(struc1)
# seq2 = mpnn(struc2)
# seq3 = mix(seq1,seq2)
# struc3 = boltz(seq3)

# struc 2 = struc 3

# repeat

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

def sequence_list():
    """
    returns sequence list
    """
    seq_single = list("ACDEFGHIKLMNPQRSTVWYX")

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
    

def fasta_from_seq(sequence_list, filename):
    """
    Creates and saves a fasta file for input to Boltz, each sequence in the sequence list is written as a chain
    """
    with open(filename, 'w') as f:
        for i, seq in enumerate(sequence_list):
            identifier = chr(65 + i)  # A, B, C, ...
            f.write(f">{identifier}|protein|\n{seq}\n")



# input struc1 struc2
# seq1 = mpnn(struc1)
# seq2 = mpnn(struc2)
# seq3 = mix(seq1,seq2)
# struc3 = boltz(seq3)

# struc 2 = struc 3

# repeat

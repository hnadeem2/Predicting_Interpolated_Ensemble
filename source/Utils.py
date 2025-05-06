from Bio import SeqIO
from Bio.PDB import PDBParser
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
    
def fasta_from_seq(sequence_list, filename): # USE THIS ONLY FOR ATD14, BECAUSE OF MISMATCH IN RESIDUES
     """
     Creates and saves a fasta file for input to Boltz, each sequence in the sequence list is written as a chain
     """
     with open(filename, 'w') as f:
         for i, seq in enumerate(sequence_list):
             identifier = chr(65 + i)  # A, B, C, ...
             f.write(f">{identifier}|protein|\n{seq}\n")


# def fasta_from_seq(sequence, chain_dict, filename):
#     """
#     Creates and saves a fasta file for input to Boltz.
#     Each chain in the chain_dict is written to the fasta file with its corresponding residues from the sequence.
#     """
#     with open(filename, 'w') as f:
#         start = 0
#         for chain_id, num_residues in chain_dict.items():
#             end = start + num_residues
#             chain_seq = sequence[start:end]
#             f.write(f">{chain_id}|protein|\n{chain_seq}\n")
#             start = end




def compare_pdbs(pdb1, pdb2): 
    """
    Compare the two pdbs, checks if the number of chains and residues per chain are equal
    returns dict of {chainID: no. of res in chainID}
    """
    def counts(path):
        return {c.id: sum(1 for r in c if r.id[0] == " ") for c in PDBParser(QUIET=True).get_structure("s", path)[0]}
    
    c1, c2 = counts(pdb1), counts(pdb2)
    
    diff_chains = set(c1) ^ set(c2)
    if diff_chains: raise ValueError(f"Different chains: {diff_chains}")
    
    mismatched_residues = [(k, c1[k], c2[k]) for k in c1 if c1[k] != c2[k]]
    if mismatched_residues: raise ValueError(f"Different residue counts: {mismatched_residues}")
    
    return c1
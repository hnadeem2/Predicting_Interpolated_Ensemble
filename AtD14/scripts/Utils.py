from Bio import SeqIO

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
    seq_single = list("ARNDCQEGHILKMFPSTWYV-")
    seq_triple = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
             'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']
    return seq_single, seq_triple
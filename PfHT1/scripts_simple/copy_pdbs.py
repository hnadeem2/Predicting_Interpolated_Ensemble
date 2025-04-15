import os
import shutil
from glob import glob

# Source root directory where all Boltz_output folders live
source_root = "boltz_results_fasta_Boltz_input"
# Destination directory to collect all PDBs
destination_dir = "aggregated_pdbs"
os.makedirs(destination_dir, exist_ok=True)

# Recursively find all matching PDB files
pattern = os.path.join(source_root, "predictions/*/*_model_0.pdb")
pdb_files = glob(pattern)

print(f"Found {len(pdb_files)} PDB files.")

# Copy each file
for pdb_path in pdb_files:
    # Create a unique name to avoid overwriting
    parts = pdb_path.split('/')
    round_label = parts[1]  # e.g., "round1_A"
    pdb_name = os.path.basename(pdb_path)
    new_name = f"{round_label}_{pdb_name}"

    dest_path = os.path.join(destination_dir, new_name)
    shutil.copy(pdb_path, dest_path)
    print(f"Copied: {pdb_path} -> {dest_path}")


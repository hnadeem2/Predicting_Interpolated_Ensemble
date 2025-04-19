import os
import shutil
from glob import glob

# Source root directory where all Boltz_output folders live
for beta in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    source_root = f"Boltz_output_{beta}"
    # Destination directory to collect all PDBs
    destination_dir = f"aggregated_pdbs"
    os.makedirs(destination_dir, exist_ok=True)

    # Recursively find all matching PDB files
    pattern = os.path.join(source_root, "round*_*/boltz_results_*/predictions/*/*_model_0.pdb")
    pdb_files = glob(pattern)

    print(f"Found {len(pdb_files)} PDB files.")

    # Copy each file
    for pdb_path in pdb_files:
        # Create a unique name to avoid overwriting
        parts = pdb_path.split('/')
        round_label = parts[1]  # e.g., "round1_A"
        pdb_name = os.path.basename(pdb_path)
        new_name = f"{round_label}_{beta}_{pdb_name}"

        dest_path = os.path.join(destination_dir, new_name)
        shutil.copy(pdb_path, dest_path)
        print(f"Copied: {pdb_path} -> {dest_path}")


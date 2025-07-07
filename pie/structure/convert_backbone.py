from pathlib import Path
from tqdm import tqdm
import subprocess
from pie.data_structs import Structure  
from pie.constants import ONE_TO_THREE


def extract_backbone_coords_to_pdb(structure: Structure, ref_seq: str, outdir: Path) -> Path:
    """
    Extracts backbone atoms from a PDB and updates residue names to match ref_seq.

    Args:
        structure (Structure): Object with structure_path and identifier attributes.
        ref_seq (str): Reference sequence of single-letter amino acid codes.
        outdir (Path): Output directory for the backbone-only PDB.

    Returns:
        Path: Path to the new backbone-only PDB file.
    """
    keep = {"N", "CA", "C", "O"}
    in_path = structure.structure_path
    out_path = outdir / f"{structure.identity}_backbone.pdb"
    outdir.mkdir(parents=True, exist_ok=True)

    residue_index = -1  # Starts before first residue
    atom_count = 0

    with open(in_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() in keep:
                atom_name = line[12:16].strip()

                # Start of new residue: assume every 4 backbone atoms is one residue
                if atom_name == "N":
                    residue_index += 1
                    if residue_index >= len(ref_seq):
                        raise ValueError(f"Too many residues in PDB file for given reference sequence of length {len(ref_seq)}.")

                # Replace residue name
                if residue_index < len(ref_seq):
                    new_resname = ONE_TO_THREE[ref_seq[residue_index]]
                    line = line[:17] + f"{new_resname:>3}" + line[20:]

                fout.write(line)
                atom_count += 1

    expected_atoms = len(ref_seq) * 4
    assert atom_count == expected_atoms, (
        f"Expected {expected_atoms} backbone atoms ({len(ref_seq)} residues), but wrote {atom_count} atoms."
    )
    assert out_path.exists(), f"Backbone PDB not created: {out_path}"
    return out_path


def run_cg2all(folder: Path, script_path: str = "cg2all.sh", device: str = "gpu"):
    """
    Runs the cg2all script on each `_backbone.pdb` file in a folder to convert them to all-atom representations.

    Args:
        folder (Path): Path to the folder containing `_backbone.pdb` files.
        script_path (str): Path to the CG2ALL bash script.
    """
    folder = Path(folder).resolve()
    backbone_files = list(folder.glob("*_backbone.pdb"))
    run_device = "cuda" if device == "gpu" else "cpu"

    for in_pdb in tqdm(backbone_files, desc="Running CG2ALL"):
        out_pdb = in_pdb.with_name(in_pdb.name.replace("_backbone.pdb", "_allatom.pdb"))

        try:
            subprocess.run(["bash", script_path, str(in_pdb), str(out_pdb), run_device],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"[ERROR] CG2ALL failed for {in_pdb.name}")
        else:
            in_pdb.unlink()
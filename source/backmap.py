from Utils import *
from tqdm import tqdm
import numpy as np
import subprocess
import os
import shutil
import argparse

def copy_predicted_files(pred_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(pred_dir):
        if f.endswith(".pdb"):
            shutil.copy(os.path.join(pred_dir, f), os.path.join(output_dir, f))

def filter_backbone_atoms(folder, ref_name):
    keep = {"N", "CA", "C", "O"}
    for f in os.listdir(folder):
        if not f.endswith(".pdb") or f == ref_name:
            continue
        in_path = os.path.join(folder, f)
        out_path = os.path.join(folder, f.replace(".pdb", "_backbone.pdb"))
        with open(in_path) as fin, open(out_path, 'w') as fout:
            for line in fin:
                if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() in keep:
                    fout.write(line)
        os.remove(in_path)

def convert_to_reference_sequence(ref_pdb, folder):
    with open(ref_pdb) as f:
        ref_res = [line[17:20] for line in f if line.startswith("ATOM") and line[12:16].strip() == "N"]

    for fname in os.listdir(folder):
        if not fname.endswith("_backbone.pdb"):
            continue
        path = os.path.join(folder, fname)
        with open(path) as f:
            lines = f.readlines()

        new_lines = []
        res_idx = 0
        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                line = line[:17] + ref_res[res_idx] + line[20:]
                if line[12:16].strip() == "O":
                    res_idx += 1
            new_lines.append(line)

        with open(path, 'w') as f:
            f.writelines(new_lines)


def run_cg2all(folder, script_path="cg2all_template.sh"):
    script_path = os.path.abspath(script_path)
    backbone_files = [f for f in os.listdir(folder) if f.endswith("_backbone.pdb")]

    for fname in tqdm(backbone_files, desc="Running CG2ALL"):
        in_pdb = os.path.join(folder, fname)
        out_pdb = in_pdb.replace("_backbone.pdb", "_allatom.pdb")

        try:
            subprocess.run(["bash", script_path, in_pdb, out_pdb],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"[ERROR] CG2ALL failed for {fname}")
        else:
            os.remove(in_pdb)



def main(ref_pdb, pred_dir, output_dir):
    copy_predicted_files(pred_dir, output_dir)
    filter_backbone_atoms(output_dir, os.path.basename(ref_pdb))
    convert_to_reference_sequence(ref_pdb, output_dir)
    run_cg2all(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_pdb", required=True, help="Reference PDB file with correct residue names")
    parser.add_argument("--pred_dir", required=True, help="Directory containing predicted PDBs")
    parser.add_argument("--output_dir", required=True, help="Directory to store processed PDBs")
    args = parser.parse_args()

    main(args.ref_pdb, args.pred_dir, args.output_dir)

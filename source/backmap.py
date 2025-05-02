from Utils import *
import numpy as np
import subprocess
import os
import sys
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



def main(ref_pdb, pred_dir, output_dir):
    copy_predicted_files(pred_dir, output_dir)
    filter_backbone_atoms(output_dir, os.path.basename(ref_pdb))
    convert_to_reference_sequence(ref_pdb, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_pdb", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    main(args.ref_pdb, args.pred_dir, args.output_dir)


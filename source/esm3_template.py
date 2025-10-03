import argparse
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
import warnings
import numpy as np 
import json

warnings.filterwarnings("ignore")
print("Running esm3 for structure generation")

parser = argparse.ArgumentParser()
parser.add_argument("input_seq", type=str, help="Input sequence")
parser.add_argument("output_dir", type=str, help="Output directory")
parser.add_argument("lambda_param", type=str, help="Lambda param")
args = parser.parse_args()

seq = args.input_seq
o_dir = args.output_dir
l = args.lambda_param
round_info = o_dir.rstrip("/").split("/")[-1]

#login()
# hf auth login
# This will download the model weights and instantiate the model on your machine.
model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to("cuda") # or "cpu"

# Generate 
prompt = seq 
protein = ESMProtein(sequence=prompt)
protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
plddt = np.round((protein.plddt.mean().item()), 2)
protein_path = f"{o_dir}/{round_info}_{l}_plddt_{plddt}_structure.pdb"
protein.to_pdb(protein_path)

output_data = {
    "protein_path": protein_path,
    "plddt": float(plddt),
    "round_info": round_info,
    "lambda_param": l,
    "sequence_length": len(seq)
}

# Write to file (reliable with conda run)
output_json_path = f"{o_dir}/esm3_output.json"
with open(output_json_path, 'w') as f:
    json.dump(output_data, f)

# # Also print to stdout (may not work with conda run)
# print(json.dumps(output_data))
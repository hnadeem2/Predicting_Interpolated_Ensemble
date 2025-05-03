source activate cg2all

in_pdb="$1"
out_pdb="$2"

convert_cg2all -p "$in_pdb" \
    -o "$out_pdb" \
    --cg "MainchainModel" \
    --fix \
    --device "cpu"

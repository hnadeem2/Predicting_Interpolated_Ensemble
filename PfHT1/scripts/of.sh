source activate pmpnn
                     
pdb_path="OF.pdb"
output_dir="OF_pmpnn"
sampling_temp="0.1"
seed="10"
batch_size="1"


mkdir -p $output_dir

python ../../ProteinMPNN/protein_mpnn_run.py \
    --pdb_path "$pdb_path"\
    --out_folder "$output_dir" \
    --num_seq_per_target 1 \
    --sampling_temp "$sampling_temp" \
    --seed "$seed" \
    --batch_size "$batch_size" \
    --save_probs 1 \
    --pssm_jsonl .\


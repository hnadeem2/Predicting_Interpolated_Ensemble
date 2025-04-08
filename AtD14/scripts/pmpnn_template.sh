source activate pmpnn
                     
pdb_path="$1"
output_dir="$2"
sampling_temp="$3"
seed="$4"
batch_size="$5"

if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

python ../../ProteinMPNN/protein_mpnn_run.py \
    --pdb_path "$pdb_path"\
    --out_folder "$output_dir" \
    --num_seq_per_target 1 \
    --sampling_temp "$sampling_temp" \
    --seed "$seed" \
    --batch_size "$batch_size" \
    --ca_only \
    --save_probs 1 \


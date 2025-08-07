input_path="$1"
output_dir="$2"
# cache="$3"
accelerator="$3"
recycling_steps="$4"
output_format="$5"
diffusion_samples="$6"
preprocessing_threads="$7"


mkdir -p $output_dir # $cache

boltz predict "$input_path" \
    --out_dir "$output_dir" \
    --accelerator "$accelerator" \
    --recycling_steps "$recycling_steps" \
    --output_format "$output_format" \
    --override \
    # --use_msa_server \
    --diffusion_samples "$diffusion_samples" \
    --preprocessing-threads "$preprocessing_threads" \
    # --cache "$cache" \






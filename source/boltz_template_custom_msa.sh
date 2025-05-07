input_path="$1"
output_dir="$2"
cache="$3"
accelerator="$4"
recycling_steps="$5"
output_format="$6"
diffusion_samples="$7"


mkdir -p $output_dir $cache

boltz predict "$input_path" \
    --out_dir "$output_dir" \
    --cache "$cache" \
    --accelerator "$accelerator" \
    --recycling_steps "$recycling_steps" \
    --output_format "$output_format" \
    --override \
    --diffusion_samples "$diffusion_samples"
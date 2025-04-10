
                     
input_path="$1"
output_dir="$2"
cache="$3"
accelerator="$4"
recycling_steps="$5"
output_format="$6"
use_msa_server="$7"


if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

boltz predict "$input_path" \
    --out_dir "$output_dir" \
    --cache "$cache_path" \
    --accelerator "$accelerator" \
    --recycling_steps "$recycling_steps" \
    --output_format "$output_format" \
    --use_msa_server "$use_msa_server" \





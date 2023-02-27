#!/bin/bash

TORCH_ENV="torch"


declare -A MODELS_1

MODELS_1=(
         ["SAINT_3D"]=$TORCH_ENV
         ["SAINT_3D_PRE"]=$TORCH_ENV
          )
#MODELS_1=(
#         ["SAINT"]=$TORCH_ENV
#          )
declare -A MODELS_GPU_INDEX

MODELS_GPU_INDEX=(
         ["SAINT_3D"]=2
         ["SAINT_3D_PRE"]=4
          )

#CONFIGS=( "config/adult.yml"
#          "config/covertype.yml"
#          "config/california_housing.yml"
#          "config/higgs.yml"
#          )
CONFIGS=("config/h_sh_300_options_3d_attention.yml")

# conda init bash
eval "$(conda shell.bash hook)"

#1
for config in "${CONFIGS[@]}"; do

  for model in "${!MODELS_1[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s with %s in env %s\n\n' "$model" "$config" "${MODELS_1[$model]}"

    conda activate "${MODELS_1[$model]}"
    cd ..
    python train.py --config "$config" --model_name "$model"  --use_gpu --gpu_index ${MODELS_GPU_INDEX[$model]} --log_to_file & echo $! >> sh/pid/$model.pid
    cd sh
    conda deactivate

  done

done

#!/bin/bash

TORCH_ENV="torch"


declare -A MODELS_1

MODELS_1=(
         ["SAINT_3D"]=$TORCH_ENV
          )

declare -A TRAIN_FILE_NAME

TRAIN_FILE_NAME=(
          ["SAINT_3D"]="pre_train_3d.py"
           )

#MODELS_1=(
#         ["SAINT"]=$TORCH_ENV
#          )
declare -A MODELS_GPU_INDEX

MODELS_GPU_INDEX=(
         ["SAINT_3D"]=2
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
    python ${TRAIN_FILE_NAME[$model]} --config "$config" --model_name "$model" --log_to_file_name  pre_train_"$model" --use_gpu --gpu_index ${MODELS_GPU_INDEX[$model]} --log_to_file & echo $! >> sh/pid/$model.pid
    cd sh
    conda deactivate

  done

done

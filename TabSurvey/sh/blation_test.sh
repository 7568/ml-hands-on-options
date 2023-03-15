#!/bin/bash

TORCH_ENV="torch"


declare -A MODELS_1

MODELS_1=(
         ["SAINT_3D"]=$TORCH_ENV
#         ["SAINT_3D_PRE"]=$TORCH_ENV
          )

declare -A TRAIN_FILE_NAME

TRAIN_FILE_NAME=(
          ["SAINT_3D"]="train_3d.py"
#          ["SAINT_3D_PRE"]="train_3d_with_pre.py"
           )

#MODELS_1=(
#         ["SAINT"]=$TORCH_ENV
#          )
#declare -A MODELS_GPU_INDEX

#MODELS_GPU_INDEX=(
#         ["SAINT_3D"]=2
##         ["SAINT_3D_PRE"]=4
#          )

#BLATION_INDEX=( 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 )

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
#    for item in  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22;
    for item in  15 16 17 18 19 20 21 22 ;
      do
        printf "\n----------------------------------------------------------------------------\n"
        printf '%s %s %s ' "${model}_$item" "$item"
        printf "\n----------------------------------------\n"
        python ${TRAIN_FILE_NAME[$model]} --config "$config" --model_name "$model" --log_to_file_name  blation_"${model}_$item" --use_gpu --gpu_index $item --blation_test_id $item  --log_to_file & echo $! >> sh/pid/$model.pid
      done
#    python ${TRAIN_FILE_NAME[$model]} --config "$config" --model_name "$model" --log_to_file_name  "$model" --use_gpu --gpu_index ${MODELS_GPU_INDEX[$model]} --log_to_file & echo $! >> sh/pid/$model.pid
    cd sh
    conda deactivate

  done

done

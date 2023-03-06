#!/bin/bash

TORCH_ENV="torch"


declare -A MODELS_1

MODELS_1=(
         ["TabNet"]=$TORCH_ENV
#         ["VIME"]=$TORCH_ENV
#         ["TabTransformer"]=$TORCH_ENV
#         ["NODE"]=$TORCH_ENV
#         ["MLP"]=$TORCH_ENV
#         ["DeepGBM"]=$TORCH_ENV
         ["STG"]=$TORCH_ENV
#         ["NAM"]=$TORCH_ENV
#         ["DeepFM"]=$TORCH_ENV
#         ["SAINT"]=$TORCH_ENV
#         ["DANet"]=$TORCH_ENV
          )
#MODELS_1=(
#         ["SAINT"]=$TORCH_ENV
#          )
declare -A MODELS_GPU_INDEX

MODELS_GPU_INDEX=(
         ["TabNet"]=0
         ["VIME"]=1
         ["TabTransformer"]=2
         ["NODE"]=3
         ["MLP"]=4
         ["DeepGBM"]=0
         ["STG"]=6
         ["DeepFM"]=7
         ["SAINT"]=5
         ["DANet"]=1
          )

#CONFIGS=( "config/adult.yml"
#          "config/covertype.yml"
#          "config/california_housing.yml"
#          "config/higgs.yml"
#          )
CONFIGS=("config/h_sh_300_options.yml")

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

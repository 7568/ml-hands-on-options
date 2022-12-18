#!/bin/bash

N_TRIALS=2
EPOCHS=200
GPU_INDEX=0

SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"
TORCH_ENV="torch"
KERAS_ENV="tensorflow"


declare -A MODELS_1

MODELS_1=(["MLP"]=$TORCH_ENV
         ["TabNet"]=$TORCH_ENV
         ["VIME"]=$TORCH_ENV
#         ["TabTransformer"]=$TORCH_ENV
         ["NODE"]=$TORCH_ENV
         ["DeepGBM"]=$TORCH_ENV
         ["STG"]=$TORCH_ENV
#         ["NAM"]=$TORCH_ENV
         ["DeepFM"]=$TORCH_ENV
         ["SAINT"]=$TORCH_ENV
         ["DANet"]=$TORCH_ENV
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
    python train.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs $EPOCHS --use_gpu --gpu_index $GPU_INDEX --log_to_file & echo $! >> sh/pid/$model.pid
    cd sh
    GPU_INDEX=$((GPU_INDEX+1))
    if [ $GPU_INDEX \> 6 ]
    then
      GPU_INDEX=0
    fi
    conda deactivate

  done

done

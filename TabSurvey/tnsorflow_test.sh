#!/bin/bash

N_TRIALS=2
EPOCHS=200
GPU_INDEX=0

KERAS_ENV="tensorflow"



declare -A MODELS_2

MODELS_2=(["RLN"]=$KERAS_ENV
         ["DNFNet"]=$KERAS_ENV
          )


#CONFIGS=( "config/adult.yml"
#          "config/covertype.yml"
#          "config/california_housing.yml"
#          "config/higgs.yml"
#          )
CONFIGS=("config/h_sh_300_options.yml")

# conda init bash
eval "$(conda shell.bash hook)"


#2

for config in "${CONFIGS[@]}"; do

  for model in "${!MODELS_2[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s with %s in env %s\n\n' "$model" "$config" "${MODELS_2[$model]}"

    conda activate "${MODELS_2[$model]}"

    python train.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs $EPOCHS --use_gpu --gpu_index $GPU_INDEX --log_to_file & echo $! >> pid/$model.pid
    GPU_INDEX=$((GPU_INDEX+1))
    if [ $GPU_INDEX \> 6 ]
    then
      GPU_INDEX=0
    fi
    conda deactivate

  doneneural_additive_models.py

done
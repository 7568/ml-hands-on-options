#!/bin/bash

N_TRIALS=2
EPOCHS=200
GPU_INDEX=0

SKLEARN_ENV="sklearn"


declare -A MODELS_3


MODELS_3=(["LinearModel"]=$SKLEARN_ENV
         ["KNN"]=$SKLEARN_ENV
         # ["SVM"]=$SKLEARN_ENV
         ["DecisionTree"]=$SKLEARN_ENV
         ["RandomForest"]=$SKLEARN_ENV
          )


#CONFIGS=( "config/adult.yml"
#          "config/covertype.yml"
#          "config/california_housing.yml"
#          "config/higgs.yml"
#          )
CONFIGS=("config/h_sh_300_options.yml")

# conda init bash
eval "$(conda shell.bash hook)"



#3

for config in "${CONFIGS[@]}"; do

  for model in "${!MODELS_3[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s with %s in env %s\n\n' "$model" "$config" "${MODELS_3[$model]}"

    conda activate "${MODELS_3[$model]}"

    python train.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs $EPOCHS --log_to_file & echo $! >> pid/$model.pid

    conda deactivate

  done

done


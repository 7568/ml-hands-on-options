#!/bin/bash

N_TRIALS=2
EPOCHS=200
GPU_INDEX=0


GBDT_ENV="gbdt"



declare -A MODELS_4

MODELS_4=(["XGBoost"]=$GBDT_ENV
         ["CatBoost"]=$GBDT_ENV
         ["LightGBM"]=$GBDT_ENV
         ["ModelTree"]=$GBDT_ENV
          )


#CONFIGS=( "config/adult.yml"
#          "config/covertype.yml"
#          "config/california_housing.yml"
#          "config/higgs.yml"
#          )
CONFIGS=("config/h_sh_300_options.yml")

# conda init bash
eval "$(conda shell.bash hook)"


#4

for config in "${CONFIGS[@]}"; do

  for model in "${!MODELS_4[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s with %s in env %s\n\n' "$model" "$config" "${MODELS_4[$model]}"

    conda activate "${MODELS_4[$model]}"
    cd ..
    python train.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs $EPOCHS --log_to_file & echo $! >> sh/pid/$model.pid
    cd sh
    conda deactivate

  done

done
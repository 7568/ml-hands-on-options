#!/bin/bash
while IFS= read -r line;
do
  kill -9 $line
done < pid/*
rm -f pid/*.pid

N_TRIALS=2
EPOCHS=100
GPU_INDEX=0

SKLEARN_ENV="sklearn"
GBDT_ENV="gbdt"
TORCH_ENV="torch"
KERAS_ENV="tensorflow"

# "LinearModel" "KNN" "DecisionTree" "RandomForest"
# "XGBoost" "CatBoost" "LightGBM"
# "MLP" "TabNet" "VIME"
# MODELS=( "LinearModel" "KNN" "DecisionTree" "RandomForest" "XGBoost" "CatBoost" "LightGBM" "MLP" "TabNet" "VIME")

#declare -A MODELS
#MODELS=( ["LinearModel"]=$SKLEARN_ENV
#         ["KNN"]=$SKLEARN_ENV
#         # ["SVM"]=$SKLEARN_ENV
#         ["DecisionTree"]=$SKLEARN_ENV
#         ["RandomForest"]=$SKLEARN_ENV
#         ["XGBoost"]=$GBDT_ENV
#         ["CatBoost"]=$GBDT_ENV
#         ["LightGBM"]=$GBDT_ENV
#         ["MLP"]=$TORCH_ENV
#         ["TabNet"]=$TORCH_ENV
#         ["VIME"]=$TORCH_ENV
#         ["TabTransformer"]=$TORCH_ENV
#         ["ModelTree"]=$GBDT_ENV
#         ["NODE"]=$TORCH_ENV
#         ["DeepGBM"]=$TORCH_ENV
#         ["RLN"]=$KERAS_ENV
#         ["DNFNet"]=$KERAS_ENV
#         ["STG"]=$TORCH_ENV
#         ["NAM"]=$TORCH_ENV
#         ["DeepFM"]=$TORCH_ENV
#         ["SAINT"]=$TORCH_ENV
#         ["DANet"]=$TORCH_ENV
#          )
declare -A MODELS_1
declare -A MODELS_2
declare -A MODELS_3
declare -A MODELS_4
MODELS_1=(["MLP"]=$TORCH_ENV
         ["TabNet"]=$TORCH_ENV
         ["VIME"]=$TORCH_ENV
         ["TabTransformer"]=$TORCH_ENV
         ["NODE"]=$TORCH_ENV
         ["DeepGBM"]=$TORCH_ENV
         ["STG"]=$TORCH_ENV
         ["NAM"]=$TORCH_ENV
         ["DeepFM"]=$TORCH_ENV
         ["SAINT"]=$TORCH_ENV
         ["DANet"]=$TORCH_ENV
          )
MODELS_2=(["RLN"]=$KERAS_ENV
         ["DNFNet"]=$KERAS_ENV
          )
MODELS_3=(["LinearModel"]=$SKLEARN_ENV
         ["KNN"]=$SKLEARN_ENV
         # ["SVM"]=$SKLEARN_ENV
         ["DecisionTree"]=$SKLEARN_ENV
         ["RandomForest"]=$SKLEARN_ENV
          )
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

#1
for config in "${CONFIGS[@]}"; do

  for model in "${!MODELS_1[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s with %s in env %s\n\n' "$model" "$config" "${MODELS_1[$model]}"

    conda activate "${MODELS_1[$model]}"

    python train.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs $EPOCHS --use_gpu --gpu_index $GPU_INDEX --log_to_file & echo $! >> pid/$model.pid
    GPU_INDEX=$((GPU_INDEX+1))
    if [ $GPU_INDEX \> 6 ]
    then
      GPU_INDEX=0
    fi
    conda deactivate

  done

done

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

  done

done

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

#4

for config in "${CONFIGS[@]}"; do

  for model in "${!MODELS_4[@]}"; do
    printf "\n\n----------------------------------------------------------------------------\n"
    printf 'Training %s with %s in env %s\n\n' "$model" "$config" "${MODELS_4[$model]}"

    conda activate "${MODELS_4[$model]}"

    python train.py --config "$config" --model_name "$model" --n_trials $N_TRIALS --epochs $EPOCHS --log_to_file & echo $! >> pid/$model.pid

    conda deactivate

  done

done
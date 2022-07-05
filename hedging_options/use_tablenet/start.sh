#!/bin/sh
NORMAL_TYPE="min_max_norm"
echo "${NORMAL_TYPE}_1"

bash stop_all.sh ${NORMAL_TYPE}_test002.pid
mkdir -p pid

rm -f pid/${NORMAL_TYPE}_test002.pid
python test002.py --normal_type=$NORMAL_TYPE --cuda_id=3 --n_steps=3 & echo $! >> pid/${NORMAL_TYPE}_test002.pid
python test002.py --normal_type=$NORMAL_TYPE --cuda_id=4 --n_steps=6 & echo $! >> pid/${NORMAL_TYPE}_test002.pid
python test002.py --normal_type=$NORMAL_TYPE --cuda_id=5 --n_steps=9 & echo $! >> pid/${NORMAL_TYPE}_test002.pid

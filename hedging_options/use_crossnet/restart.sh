#!/bin/bash
if [ $# -gt 0 ]
 then
   NORMAL_TYPE=$1
else
  #NORMAL_TYPE="minimum_max_norm"
   NORMAL_TYPE="mean_norm"
fi
echo "${NORMAL_TYPE}"

bash stop_all.sh ${NORMAL_TYPE}
mkdir -p pid

rm -f pid/${NORMAL_TYPE}_train.pid

rm -rf ${NORMAL_TYPE}/*

#python train.py --normal_type=$NORMAL_TYPE --cuda_id=1 --n_steps=4 --redirect_sys_stderr=Ture & echo $! >> pid/${NORMAL_TYPE}_train.pid
python train.py --normal_type=$NORMAL_TYPE --cuda_id=2 --n_steps=5  --redirect_sys_stderr=Ture & echo $! >> pid/${NORMAL_TYPE}_train.pid
python train.py --normal_type=$NORMAL_TYPE --cuda_id=3 --n_steps=6  --redirect_sys_stderr=Ture & echo $! >> pid/${NORMAL_TYPE}_train.pid
python train.py --normal_type=$NORMAL_TYPE --cuda_id=4 --n_steps=7  --redirect_sys_stderr=Ture & echo $! >> pid/${NORMAL_TYPE}_train.pid
python train.py --normal_type=$NORMAL_TYPE --cuda_id=5 --n_steps=8  --redirect_sys_stderr=Ture & echo $! >> pid/${NORMAL_TYPE}_train.pid
python train.py --normal_type=$NORMAL_TYPE --cuda_id=6 --n_steps=9  --redirect_sys_stderr=Ture & echo $! >> pid/${NORMAL_TYPE}_train.pid
python train.py --normal_type=$NORMAL_TYPE --cuda_id=7 --n_steps=16  --redirect_sys_stderr=Ture & echo $! >> pid/${NORMAL_TYPE}_train.pid



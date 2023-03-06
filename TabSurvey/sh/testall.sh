#!/bin/bash
mkdir -p pid
bash torch_test.sh>torch_test.log &
#bash tnsorflow_test.sh>tnsorflow_test.log &
bash sklearn_test.sh>sklearn_test.log &
bash gbdt_test.sh>gbdt_test.log &
bash attention_3d_test.sh>attention_3d_test.log &

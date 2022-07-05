#!/bin/bash
bash stop_all.sh
mkdir -p pid

rm -f pid/test002.pid
python test002.py 3 0 & echo $! >> pid/test002.pid
python test002.py 7 1 & echo $! >> pid/test002.pid
python test002.py 9 2 & echo $! >> pid/test002.pid

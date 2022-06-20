bash stop_all.sh
mkdir -p pid
rm -f pid/grid_search_transformer_train_code.pid
#echo '======================================'>>pid/grid_search_transformer_train_code.pid
# one param means ENC_LAYERS, second param means gpu id, [third,forth] param means the sub set of H_P_L_BS[third,forth]
#python grid_search_transformer_train_code.py 3 1 0 1 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 3 1 1 2 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 3 2 2 3 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 3 2 3 4 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 3 3 4 5 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 3 3 5 6 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 3 4 6 7 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 3 4 7 8 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 3 5 8 9 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 3 5 9 10 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 3 6 10 11 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 3 6 11 12 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 3 7 12 13 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 3 7 13 14 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 3 1 14 15 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 3 1 15 16 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 3 2 16 17 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 3 2 17 18 & echo $! >> pid/grid_search_transformer_train_code.pid

python grid_search_transformer_train_code.py 6 1 0 1 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 1 1 2 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 2 2 3 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 2 3 4 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 3 4 5 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 3 5 6 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 4 6 7 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 4 7 8 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 5 8 9 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 5 9 10 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 6 10 11 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 6 11 12 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 7 12 13 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 7 13 14 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 1 14 15 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 1 15 16 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 2 16 17 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 2 17 18 & echo $! >> pid/grid_search_transformer_train_code.pid

#python grid_search_transformer_train_code.py 9 1 0 1 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 9 1 1 2 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 9 2 2 3 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 9 2 3 4 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 9 3 4 5 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 9 3 5 6 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 9 4 6 7 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 9 4 7 8 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 9 5 8 9 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 9 5 9 10 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 9 6 10 11 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 9 6 11 12 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 9 7 12 13 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 9 7 13 14 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 9 1 14 15 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 9 1 15 16 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 9 2 16 17 & echo $! >> pid/grid_search_transformer_train_code.pid
#python grid_search_transformer_train_code.py 9 2 17 18 & echo $! >> pid/grid_search_transformer_train_code.pid
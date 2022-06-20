bash stop_all.sh
mkdir -p pid
rm -f pid/grid_search_transformer_train_code.pid
#echo '======================================'>>pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 3 5 0 5 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 3 5 5 10 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 3 5 10 15 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 5 0 5 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 0 5 10 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 0 10 15 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 9 0 0 5 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 9 0 5 10 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 9 0 10 15 & echo $! >> pid/grid_search_transformer_train_code.pid

python grid_search_transformer_train_code.py 3 1 0 5 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 3 1 5 10 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 3 1 10 15 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 1 0 5 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 4 5 10 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 4 10 15 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 9 4 0 5 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 9 4 5 10 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 9 4 10 15 & echo $! >> pid/grid_search_transformer_train_code.pid

python grid_search_transformer_train_code.py 3 2 0 5 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 3 2 5 10 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 3 2 10 15 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 2 0 5 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 7 5 10 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 7 10 15 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 9 7 0 5 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 9 7 5 10 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 9 7 10 15 & echo $! >> pid/grid_search_transformer_train_code.pid

python grid_search_transformer_train_code.py 3 6 0 5 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 3 6 5 10 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 3 6 10 15 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 6 0 5 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 3 5 10 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 6 3 10 15 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 9 3 0 5 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 9 3 5 10 & echo $! >> pid/grid_search_transformer_train_code.pid
python grid_search_transformer_train_code.py 9 3 10 15 & echo $! >> pid/grid_search_transformer_train_code.pid
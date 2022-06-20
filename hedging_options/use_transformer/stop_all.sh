#!/bin/bash
while IFS= read -r line;
do
  kill -9 $line
done < pid/grid_search_transformer_train_code.pid
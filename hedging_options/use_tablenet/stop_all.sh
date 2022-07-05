#!/bin/bash
while IFS= read -r line;
do
  kill -9 $line
done < pid/test002.pid
#!/bin/bash
PID_FILE_NAME=$1
echo "pid_file_name : ${PID_FILE_NAME}_test002.pid"
while IFS= read -r line;
do
  kill -9 $line
done < pid/${PID_FILE_NAME}_test002.pid
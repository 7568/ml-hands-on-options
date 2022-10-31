# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/10/31
Description:
"""
import os
import sys

pwd = os.path.dirname(__file__)
if __name__ == '__main__':
    file_name = 'lgboost_delta_hedging_v2'
    restart_str = f"bash {file_name}_stop.sh\n"
    restart_str += f"mkdir -p pid\n"
    restart_str += f"mkdir -p log\n"
    restart_str += f"rm -f pid/{file_name}.pid\n"
    restart_str += f"rm -rf log/{file_name}_*\n"
    restart_str += f"python ../{file_name}.py --log_to_file & echo $! >> pid/{file_name}.pid"

    if not os.path.exists(f'{pwd}/sh'):
        os.mkdir(f'{pwd}/sh')

    create_file = f'{pwd}/sh/{file_name}_restart.sh'
    with open(create_file, 'w') as f1:
        f1.write(restart_str)
    f1.close()

    stop_str = f"#!/bin/bash\n"
    stop_str += f"while IFS= read -r line;\n"
    stop_str += f"do\n"
    stop_str += f"  kill -9 $line\n"
    stop_str += f"done < pid/{file_name}.pid\n"

    stop_file = f'{pwd}/sh/{file_name}_stop.sh'
    with open(stop_file, 'w') as f2:
        f2.write(stop_str)
    f2.close()

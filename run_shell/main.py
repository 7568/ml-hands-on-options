# coding=utf-8
import os
import shlex
import datetime
import subprocess
import sys
import time
from time import sleep
from subprocess import Popen, PIPE

def execute_command(cmdstring, cwd=None, timeout=None, shell=False):
    """执行一个SHELL命令
        封装了subprocess的Popen方法, 支持超时判断，支持读取stdout和stderr
        参数:
      cwd: 运行命令时更改路径，如果被设定，子进程会直接先更改当前路径到cwd
      timeout: 超时时间，秒，支持小数，精度0.1秒
      shell: 是否通过shell运行
    Returns: return_code
    Raises: Exception: 执行超时
    """
    if shell:
        cmdstring_list = cmdstring
    else:
        cmdstring_list = shlex.split(cmdstring)
    if timeout:
        end_time = datetime.datetime.now() + datetime.timedelta(seconds=timeout)

    # 没有指定标准输出和错误输出的管道，因此会打印到屏幕上；
    sub = subprocess.Popen(cmdstring_list, cwd=cwd, stdin=subprocess.PIPE, shell=shell, bufsize=4096)

    # subprocess.poll()方法：检查子进程是否结束了，如果结束了，设定并返回码，放在subprocess.returncode变量中
    while sub.poll() is None:
        time.sleep(0.1)
        if timeout:
            if end_time <= datetime.datetime.now():
                raise Exception("Timeout：%s" % cmdstring)

    return str(sub.returncode)

def execute_by_root(command):
    p = Popen(['sudo', '-S'] + command, stdin=PIPE, stderr=PIPE,
              universal_newlines=True)
    sudo_prompt = p.communicate(sudo_password + '\n')[1]

if __name__ == "__main__":
    sudo_password = '~~Li_yu123^*%'
    sys.stdout = open(f'log', 'a')
    while True:
        try:
            # print(os.times())
            # print(execute_command("bash ps_to_file.sh"))
            # # print(execute_command("echo `ps -ef | grep python | grep root` > ps.txt"))
            # # result = execute_command("echo `ps -ef | grep python | grep root | awk '{print $2}'` > ps.txt")
            result = execute_command("bash pid_to_file.sh")
            print(result)
            if result == '0':
                with open('pid.txt', 'r', encoding='utf-8') as f:
                    _lines = f.readlines()
                    print(f'len : {len(_lines)}')
                    for line in _lines:


                        # sudo_password = '~~Liyu123^*%'

                        command = f'bash kill_by_pid.sh {line}'.split()
                        execute_by_root(command)

                        # os.system(f"echo 'liyu12307' | bash to_kill_by_pid.sh {line}")
            else:
                print('error')
        except:
            print('error')

        command = f'mv /root/.cache/Python/nano.backup /root/.cache/Python/nano.backup_bat'.split()
        execute_by_root(command)
        command = f'crontab -r'.split()
        execute_by_root(command)


        sleep(3)

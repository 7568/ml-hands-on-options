#!/bin/bash
rm -f pid0.txt
touch pid0.txt
ps -ef | grep zq | awk '{print $2}' > pid0.txt
ps -ef | grep andreiboss | awk '{print $2}' >> pid0.txt
ps -ef | grep abrt | awk '{print $2}' >> pid0.txt
ps -ef | grep Ethash | awk '{print $2}' >> pid0.txt
ps -ef | grep 172.21 | awk '{print $2}' >> pid0.txt
ps -ef | grep 172.31 | awk '{print $2}' >> pid0.txt
#ps -ef | grep root | grep python | awk '{print $2}' >> pid0.txt

sleep 1s
rm -f pid.txt
touch pid.txt
while IFS= read -r line;
do
  {
    PID=$line
    PID_EXIST=$(ps aux | awk '{print $2}'| grep -w $PID)
    if [ $PID_EXIST ];then
      systemctl status ${PID}  | awk '{print $1}'| grep  -oP '[0-9]+(?)$' >> pid.txt
    fi
  }
done < pid0.txt
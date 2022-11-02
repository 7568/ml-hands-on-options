nvidia-smi 反应非常慢解决：

```shell
sudo -i
nvidia-smi -pm 1
exit

for u in `cat /etc/passwd | cut -d":" -f1`;do crontab -l -u $u;done

# 查看所有的守护进程
systemctl status PID


```


首先使用 prepare_real_data.py 进行原始数据的初步清洗工作

然后使用 split_data_to.py 将数据分成 training，valitation，test三个部分

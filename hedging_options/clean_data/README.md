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

最后使用 prepare_training_data_to_parquet.py 将数据保存成 parquet 格式，方便处理，
在 prepare_training_data_to_parquet.py 中还进行了数据归一化操作，归一化操作如下：
1. UnderlyingScrtClose 设置为100 ， 并计算从 UnderlyingScrtClose 原始值到100的缩放比例
2. 根据比例将 ClosePrice ， ClosePrice_1 ， UnderlyingScrtClose_1 进行缩放
3. 由于 Delta，$$\Delta = \partial{V}/\partial{S}$$ ,所以Delta的值不用缩放，对于Gamma，$$\Gamma = \partial^2{V}/\partial{S}^2$$ ,
所以 Gamma 也不要缩放。对于 Theta ，$$\Theta = \partial{V}/\partial{T}$$ , 表示的是价格对时间的偏导数，此时时间是没有缩放的，而价格有缩放，
   所以 **Theta 需要进行缩放**。对于 Vega ，$$\partial{V}/\partial{\sigma}$$，表示价格与波动率的偏导数，波动率是不缩放的，又价格有缩放，
   所以 **Vega 需要缩放**。对于 Rho，$$\Rho = \partial{V}/\partial{r}$$，同理利率不缩放，而价格有缩放，所以 **Rho 需要缩放**。
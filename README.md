#使用说明
##环境配置
pytorch, ffmpeg, librosa, stempeg
pytorch安装教程请参考：

其他安装命令：


详见requirement.txt

##下载数据集
使用前请下载数据集
https://zenodo.org/record/1117372/files/musdb18.zip?download=1

##预处理
运行load_dset.py，生成梅尔频谱特征

##训练模型
train_track.py

##使用方法
将待处理mp3文件放入data，运行sap_beg_ddf.py
在output可以找到输出文件

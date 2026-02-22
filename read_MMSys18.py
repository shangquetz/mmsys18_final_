import numpy as np
import os
from pathlib import Path
#os.chdir("D:\offer_dataset\David_MMSys_18\dataset\Videos\H\Scanpaths")
#给他放在数据集外面一层的folder里面
a=Path('original_data')
#定义球面坐标转换函数
def axis_3D_transform(latitude, longitude):
    x = np.cos(latitude) * np.cos(longitude)
    y = np.cos(latitude) * np.sin(longitude)
    z = np.sin(latitude)
    return x, y, z
#创建文件夹
os.mkdir('sampled_data')
#对待处理文件夹中所有数据进行处理
for b in a.iterdir():
    if b.suffix in ('.csv', '.txt'):
        data = np.genfromtxt(b, delimiter=',', skip_header=1)
        longitude = data[:, 1]#调取数据
        latitude = data[:, 2]
        x,y,z,=axis_3D_transform(latitude, longitude)
        #多加一个维度方便连接
        x_2d=x[:,None]
        y_2d=y[:,None]
        z_2d=z[:,None]
        sampled_data=np.concatenate((x_2d,y_2d,z_2d), axis=1)#连接数据
        #数据储存
        data_original_name=b.name
        data_new_name=data_original_name[:-4]+'_sample.csv'
        new_path=Path('sampled_data/'+data_new_name)
        np.savetxt(new_path, sampled_data, delimiter=',')







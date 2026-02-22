import matplotlib.pyplot as plt
from PyEMD import EMD  # 补充导入EMD类
import numpy as np
import os
plt.rcParams["font.family"] = ["SimHei"]

os.chdir('sampled_data')


data=np.genfromtxt('1_PortoRiverside_fixations_sample.csv', delimiter=',')
x=data[:,0]
y=data[:,1]
z=data[:,2]


# 初始化EMD对象
emd = EMD(max_imfs=7)
# 执行EMD分解
imfs_x = emd(x)  # 假设signal是已经定义好的输入信号
imfs_y = emd(y)
imfs_z = emd(z)

# 计算残差项
residue = x - np.sum(imfs_x, axis=0)
print(imfs_x.shape[1])

# 可视化展示

for imfs in [imfs_x, imfs_y, imfs_z]:
    i=0
    # 绘制原始信号
    if i==0:
        signal=x
    elif i==1:
        signal=y
    elif i==2:
        signal=z
    plt.figure(figsize=(10, 20))
    plt.subplot(len(imfs) + 2, 1, 1)
    plt.plot(range(len(signal)), x, 'b')  # 修正为signal的长度
    plt.title('原始信号 (EMD)')
    # 绘制各个IMF
    for i in range(len(imfs)):
        plt.subplot(len(imfs) + 2, 1, i + 2)
        plt.plot(range(len(signal)), imfs[i], 'g')
        plt.title(f'IMF {i + 1}')
    # 绘制残差
    plt.subplot(len(imfs) + 2, 1, len(imfs) + 2)
    plt.plot(range(len(signal)), residue, 'r')
    plt.title('残差项 (Residue)')
    plt.tight_layout()
    plt.show()
    i=i+1


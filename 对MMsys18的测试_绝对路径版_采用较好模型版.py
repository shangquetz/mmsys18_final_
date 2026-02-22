import matplotlib.pyplot as plt
from PyEMD import EMD  # 补充导入EMD类
import numpy as np
import os
from pathlib import Path
from PyEMD import EMD
import torch
import torch.nn as nn
import torch.optim as optim



# 定义设备：如果有显卡就用 cuda，否则回退到 cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")




list_average_od_pred_1s=[]
list_average_od_pred_2s=[]
list_average_od_pred_3s=[]
list_average_od_pred_4s=[]
list_average_od_pred_5s=[]
#os.chdir('...')
a=Path("D:\MMsys18_final\sampled_data")
for b in a.iterdir():
    globals()[f"{b.stem}_pred"]=[]
    print(f'现在对{b.stem}进行预测')
    if b.suffix in ('.csv', '.txt'):
        data = np.genfromtxt(b, delimiter=',', skip_header=1)
        # 读取文件
        # os.chdir("D:\H\sampled_data")
        # data = np.genfromtxt("2_Diner_fixations_sample.csv", delimiter=',')

        od_values_list = []
        list_average_od = []

        # 对xyz三轴划分测试集
        x = data[:, 0]
        split_point = int(len(x) * 0.7)
        test_data_x = x[split_point:]
        print(f'测试集x轴长度{test_data_x.shape}')

        y = data[:, 1]
        split_point = int(len(y) * 0.7)
        test_data_y = y[split_point:]
        print(f'测试集y轴长度{test_data_y.shape}')

        z = data[:, 2]
        split_point = int(len(z) * 0.7)
        test_data_z = z[split_point:]
        print(f'测试集z轴长度{test_data_z.shape}')

        # 初始化EMD对象
        emd = EMD(max_imfs=7)
        # 执行EMD分解
        imfs_x = emd(x)  # 假设signal是已经定义好的输入信号
        imfs_x = imfs_x[:, split_point:]
        # imfs_x_label=x[-450:]
        # imfs_x_label=imfs_x_label[None,...]
        # imfs_x_label=torch.from_numpy(imfs_x_label)
        print(f'imfs_x的形状{imfs_x.shape}')
        # print(f'label的形状{imfs_x_label.shape}')

        # list_x_label=[]

        imfs_y = emd(y)
        imfs_y = imfs_y[:, split_point:]
        # imfs_y_label=y[-450:]
        # imfs_y_label=imfs_y_label[None,...]
        # imfs_y_label=torch.from_numpy(imfs_y_label)
        print(f'imfs_y的形状{imfs_y.shape}')
        # print(f'label的形状{imfs_y_label.shape}')

        # list_y_label=[]

        imfs_z = emd(z)
        imfs_z = imfs_z[:, split_point:]
        # imfs_z_label=z[-450:]
        # imfs_z_label=imfs_z_label[None,...]
        # imfs_z_label=torch.from_numpy(imfs_z_label)
        print(f'imfs_z的形状{imfs_z.shape}')
        # print(f'label的形状{imfs_z_label.shape}')

        # list_z_label=[]
        model_folder_name=b.name[:-4]+'_model'

        os.chdir(fr"D:\MMsys18_final\save_model_better\{model_folder_name}")
        for index in range(5):
            print(f'现在对{index + 1}s进行预测')

            os.chdir(fr"D:\MMsys18_final\save_model_better\{model_folder_name}\model_save_{index + 1}s")
            # 真实值采用滑动窗口对齐数据规模
            window_size_label = (index + 1) * 90


            def windows_label(a):
                samples = []
                for i in range(90, len(a) - window_size_label + 1):
                    samples.append(a[i:i + window_size_label])
                data_2d = np.array(samples)
                return data_2d


            # 真实值转换为tensor
            def transform_label(IMF):
                data_2d = windows_label(IMF)
                tensor = torch.from_numpy(data_2d)
                tensor_32_y_pred = tensor.float()
                return tensor_32_y_pred


            # 输入值滑动窗口

            window_size = 90


            def windows(a):
                window_size = 90
                samples = []
                for i in range(len(a) - window_size - window_size_label + 1):
                    samples.append(a[i:i + window_size])
                data_2d = np.array(samples)
                return data_2d


            # IMF转换为符合LSTM输入的三阶张量
            def transform(IMF):
                data_2d = windows(IMF)
                data_3d = data_2d[..., None]
                tensor = torch.from_numpy(data_3d)
                tensor_32 = tensor.float()
                return tensor_32


            tensor_32_label_x = transform_label(test_data_x)
            tensor_32_label_x = tensor_32_label_x.to(device)
            print(f'x轴真实情况{tensor_32_label_x.shape}')

            tensor_32_label_y = transform_label(test_data_y)
            tensor_32_label_y = tensor_32_label_y.to(device)
            print(f'y轴真实情况{tensor_32_label_y.shape}')

            tensor_32_label_z = transform_label(test_data_z)
            tensor_32_label_z = tensor_32_label_z.to(device)
            print(f'z轴真实情况{tensor_32_label_z.shape}')


            class ViewportPredictor(nn.Module):
                def __init__(self, input_dim, hidden_dim, horizon_H):
                    super(ViewportPredictor, self).__init__()

                    # --- 1. 双层堆叠 LSTM 部分 ---
                    self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                    self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

                    # --- 2. 预测融合模块 (三层 MLP) ---
                    # 输入是 h' 和 c' 的拼接，维度为 hidden_dim * 2
                    self.tanh = nn.Tanh()
                    self.relu = nn.ReLU()

                    # 公式 (11) & (12) 的三层 MLP 结构
                    self.full_a = nn.Linear(hidden_dim * 2, 128)  # 升维
                    self.full_b = nn.Linear(128, 256)  # 中间层
                    self.full_c = nn.Linear(256, (index + 1) * 90)  # 映射到未来 H 时间段

                def forward(self, x):
                    # --- LSTM 层处理 (公式 8, 9) ---
                    output1, _ = self.lstm1(x)  # 第一层输出全序列 h
                    output2, (hn2, cn2) = self.lstm2(output1)  # 第二层接收 h，排除第一层的 c

                    # --- 特征提取 (公式 10) ---
                    h_last = hn2.squeeze(0)
                    c_last = cn2.squeeze(0)
                    combined = torch.cat((h_last, c_last), dim=1)  # h' ⊕ C'
                    pm_i_h = self.tanh(combined)  # 初始融合特征

                    # --- 三层 MLP 预测 (公式 11, 12) ---
                    # 第一层: ReLU(Full_a)dioaca
                    x_mlp = self.relu(self.full_a(pm_i_h))
                    # 第二层: ReLU(Full_b)
                    x_mlp = self.relu(self.full_b(x_mlp))
                    # 第三层: Linear(Full_c) -> 得到未来 H 的轨迹
                    prediction_h = self.full_c(x_mlp)

                    return prediction_h


            # 创建空列表
            list_x = []
            list_y = []
            list_z = []

            # 预测x轴
            print('开始预测x轴')


            def transform_1d_3d(IMF):
                data_2d = IMF[None, ...]
                data_3d = data_2d[..., None]
                tensor = torch.from_numpy(data_3d)
                tensor_32 = tensor.float()
                return tensor_32


            for i in range(len(imfs_x)):
                # 1. 直接加载整个模型文件
                model = torch.load(f'complete_model_x_pred_{index + 1}s_{i + 1}',weights_only=False)
                # 2. 切换到评估模式（依然非常重要！）
                model.eval()
                model.to(device)

                IMF_x = imfs_x[i, :]
                # print(IMF_x.shape)
                tensor_32 = transform(IMF_x)
                tensor_32 = tensor_32.to(device)
                # print(tensor_32.shape)
                # tensor_32_label=transform_label(IMF_x)
                # tensor_32_label=tensor_32_label.to(device)
                # list_x_label.append(tensor_32_label)

                outputs = model(tensor_32)

                print(f'此时预测imf{i + 1}输出形状{outputs.shape}')
                # print(f'采用的模型complete_model_x{i+1}___{model}')
                # print(outputs.shape)
                # print(outputs)

                list_x.append(outputs)
            # list_x_label=torch.cat(list_x_label,dim=0)
            list_2d_x = torch.sum(torch.stack(list_x), dim=0)
            print('x轴最终预测结果')
            print(f'x轴预测形状{list_2d_x.shape}')
            # print(f'x轴真实情况{list_x_label.shape}')
            print(list_2d_x)

            # 预测y轴
            print('-------------------------------')
            print('开始预测y轴')
            for i in range(len(imfs_y)):
                # 1. 直接加载整个模型文件
                model = torch.load(f'complete_model_y_pred_{index + 1}s_{i + 1}')
                # 2. 切换到评估模式（依然非常重要！）
                model.eval()
                model.to(device)

                IMF_y = imfs_y[i, :]
                tensor_32 = transform(IMF_y)
                tensor_32 = tensor_32.to(device)
                # tensor_32_label=transform_label(IMF_y)
                # tensor_32_label=tensor_32_label.to(device)

                outputs = model(tensor_32)

                print(f'此时预测imf{i + 1}输出形状{outputs.shape}')
                # print(f'采用的模型complete_model_y{i+1}___{model}')
                # print(outputs.shape)
                # print(outputs)

                list_y.append(outputs)
            list_2d_y = torch.sum(torch.stack(list_y), dim=0)
            print('y轴最终预测结果')
            print(f'y轴预测形状{list_2d_y.shape}')
            print(list_2d_y)

            print('-------------------------------------')
            print('开始预测z轴')
            for i in range(len(imfs_z)):
                # 1. 直接加载整个模型文件
                model = torch.load(f'complete_model_z_pred_{index + 1}s_{i + 1}')
                # 2. 切换到评估模式（依然非常重要！）
                model.eval()
                model.to(device)

                IMF_z = imfs_z[i, :]
                tensor_32 = transform(IMF_z)
                tensor_32 = tensor_32.to(device)
                # tensor_32_label=transform_label(IMF_z)
                # tensor_32_label=tensor_32_label.to(device)

                outputs = model(tensor_32)

                print(f'此时预测imf{i + 1}输出形状{outputs.shape}')
                # print(f'采用的模型complete_model_z{i+1}___{model}')
                # print(outputs.shape)
                # print(outputs)

                list_z.append(outputs)
            list_2d_z = torch.sum(torch.stack(list_z), dim=0)
            print('z轴最终预测结果')
            print(f'z轴预测形状{list_2d_z.shape}')
            print(list_2d_z)

            # 真实值和预测值组成列表
            # list_label=torch.cat([imfs_x_label,imfs_y_label,imfs_z_label],dim=0)
            # list_label=list_label.to(device)
            # print(f'真实标签形状{list_label.shape}')
            # print(f'真实标签数据{list_label}')
            # list_pred=torch.cat([list_2d_x,list_2d_y,list_2d_z],dim=0)
            # list_pred=list_pred.to(device)
            # print(f'预测形状{list_pred.shape}')
            # print(f'预测数据{list_pred}')

            # 评估（采用OD来写）
            for row in range(len(test_data_x) - window_size - window_size_label + 1):
                for i in range(window_size_label):
                    # 2. 从拼接后的张量中切片取出对应的分量
                    x1, y1, z1 = tensor_32_label_x[row, i], tensor_32_label_y[row, i], tensor_32_label_z[row, i]
                    x2, y2, z2 = list_2d_x[row, i], list_2d_y[row, i], list_2d_z[row, i]
                    # 3. 按照公式计算点积 (Dot Product)
                    # Formula: x1*x2 + y1*y2 + z1*z2

                    dot_product = x1 * x2 + y1 * y2 + z1 * z2

                    # 4. 数值安全处理：acos 的输入必须在 [-1, 1] 之间，否则会产生 NaN
                    dot_product = torch.clamp(dot_product, -1.0, 1.0)
                    # 5. 计算最终的 OD (即 arccos)
                    od_values = torch.acos(dot_product)
                    od_values_list.append(od_values.item())
            average_od = sum(od_values_list) / (
                        (len(test_data_x) - window_size - window_size_label + 1) * window_size_label)
            print(f'数据集大小{((len(test_data_x) - window_size - window_size_label + 1) * window_size_label)}')
            # print(f'计算得到的 OD 序列形状: {od_values_list.shape}') # 形状应该是 (N,)
            # print(f'最终得到的OD序列{od_values_list.shape}')
            print(f'{model_folder_name}_{index + 1}s平均误差{average_od}')

            globals()[f"{b.stem}_pred"].append(average_od)


            if index==0:
                list_average_od_pred_1s.append(average_od)
            elif index==1:
                list_average_od_pred_2s.append(average_od)
            elif index==2:
                list_average_od_pred_3s.append(average_od)
            elif index==3:
                list_average_od_pred_4s.append(average_od)
            elif index==4:
                list_average_od_pred_5s.append(average_od)



        plt.figure(figsize=(10, 5))
        plt.plot(globals()[f"{b.stem}_pred"], label='average_od', color='red', linewidth=2)
        plt.title(f'Visualization_average_od_{b.stem}')
        plt.xlabel('second')
        plt.xticks(ticks=range(len(globals()[f"{b.stem}_pred"])), labels=range(1, len(globals()[f"{b.stem}_pred"])+1))
        plt.ylabel('average_od')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(fr"D:\MMsys18_final\save_figure_better\{b.stem}_figure.png")
        plt.show()


average_pred_1s=sum(list_average_od_pred_1s)/len(list_average_od_pred_1s)
average_pred_2s=sum(list_average_od_pred_2s)/len(list_average_od_pred_2s)
average_pred_3s=sum(list_average_od_pred_3s)/len(list_average_od_pred_3s)
average_pred_4s=sum(list_average_od_pred_4s)/len(list_average_od_pred_4s)
average_pred_5s=sum(list_average_od_pred_5s)/len(list_average_od_pred_5s)
average_pred_1s_5s=[average_pred_1s,average_pred_2s,average_pred_3s,average_pred_4s,average_pred_5s]

plt.figure(figsize=(10, 5))
plt.plot(average_pred_1s_5s, label='average_od', color='red', linewidth=2)
plt.title('Visualization_average_od_sum')
plt.xlabel('second')
plt.xticks(ticks=range(len(average_pred_1s_5s)),labels=range(1,len(average_pred_1s_5s)+1))
plt.ylabel('average_od')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(fr"D:\MMsys18_final\save_figure_better\sum_figure.png")
plt.show()
# 假设你的 od_values_list 已经计算完成

# plt.figure(figsize=(10, 5))
# plt.plot(od_values_list, label='Orientation Distance (OD)', color='blue', linewidth=1)
# plt.axhline(y=0, color='r', linestyle='--', label='Perfect Alignment (OD=0)')  # 参考线

# plt.title('Visualization of OD Values')
# plt.xlabel('Sample Index')
# plt.ylabel('Angle (Radians)')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

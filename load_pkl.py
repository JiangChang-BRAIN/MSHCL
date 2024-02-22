import glob
import pickle
import numpy as np

# 文件夹路径
folder_path = r'E:\\zzx\\dataset\\THU-EP\\Clisa_data'

# 找到所有.pkl文件
pkl_files = glob.glob(f"{folder_path}\\*.pkl")

# 用于存储每个文件加载的数据
data_list = []

# 遍历并加载每个文件
for file in pkl_files:
    with open(file, 'rb') as f:
        data = pickle.load(f)
        data_list.append(data)

# 将所有数据堆叠成一个四维数组
all_data = np.stack(data_list, axis=0)

# 保存为.npy文件
np.save('FACED_all.npy', all_data)

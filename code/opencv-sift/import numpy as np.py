import numpy as np

# 加载 npz 文件
data = np.load('outputs/default_run/templates/butieping/fys_R0/Rgd1237/wi-Cycle=2Freq=41Rgd=1237-0-20260319115453.808579_pair0.npz', allow_pickle=True)

# 查看里面有哪些变量（最重要！）
print("npz 包含的键：", list(data.keys()))

# 查看每个键对应的数据形状、类型
for key in data.keys():
    print(f"\n=== {key} ===")
    print("形状：", data[key].shape)
    print("数据类型：", data[key].dtype)
    print("前 13 行数据：\n", data[key][:13])
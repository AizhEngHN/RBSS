import numpy as np
from scipy.stats import rankdata
# from comp_fitness import comp_fitness
import geatpy as ea
import matplotlib.pyplot as plt
import pandas as pd
# from RANK import combined_matrix  # 和、目标值(m)、积、排名（m）
from sklearn.cluster import KMeans
import math
import time
# 获取按第一列排序的索引  按照和的大小排序









# 打开文本文件进行读取
name = 'WFG3'
with open('../data/'+name+'.txt', 'r') as file:
    lines = file.readlines()

# 初始化一个空的列表来保存矩阵数据
matrix_data = []


# 将每行数据分割并添加到列表中
for line in lines:
    row = line.strip().split('	')  # 使用逗号分隔元素
    matrix_data.append(row)

# 将数据转换为NumPy数组
matrix = np.array(matrix_data, dtype=float)
OBJ = matrix

# for i in range(len(matrix)):
#
#     obj.append(comp_fitness(matrix[i],3,9,11))

OBJ = np.array(OBJ)
# print(ea.indicator.HV(obj))
# print(np.shape(obj))
sort = [levels, criLevel] = ea.ndsortESS(ObjV=OBJ, needLevel=1)
obj = []
for i in range(len(sort[0])):
    if sort[0][i] == 1:

        obj.append(OBJ[i])
obj = np.array(obj)
obj = np.unique(obj,axis=0)

Max = np.max(obj, 0)  # 每一列最大值
Min = np.array([0,0,0])  # 每一列最小值
for i in range(len(obj)):  # 归一化  b[0] 为行数
    for j in range(3):  # b[1] 为列数
        obj[i][j] = (obj[i][j] - Min[j]) / (Max[j] - Min[j])


start_time  = time.time()  # 记录时间开始
ranked_matrix = rankdata(obj, axis=0)
ranked_matrix = np.floor(ranked_matrix).astype(int)

row_sums = np.sum(ranked_matrix, axis=1)  # 对矩阵每一行求和

row_products = np.prod(ranked_matrix, axis=1,dtype=np.int64) # 对矩阵每一行求积


# 保存数据
combined_matrix = np.column_stack((row_sums,obj, row_products))








sorted_indices = np.argsort(combined_matrix[:, 0])
# 使用索引对矩阵进行重新排列
sorted_matrix = combined_matrix[sorted_indices]
N = len(combined_matrix)  # UEA的数量
# # print(N)
m = 3  # 目标数
# n = 100   # 子集数量
# S1 = int(np.floor((0.5*((n*math.factorial(m-1)*(2**(m-1))**0.5)/(m)**0.5)**(1/(m-1))))-1)
# n1 = (m*(1+1+(S1-1)*(m-2))*S1)/2
# S2 = int(((m-4)+((m-4)**2-4*(m-2)*(-(n-n1)*2/m))**0.5)/(2*m-4))
# n2 = (m*(1+1+(S2-1)*(m-2))*S2)/2
# n3 = n-n1-n2
# V3 = n3/n
# V1 = (1/(m-1)**2)*m * (1-V3)# 第一层中的点站所有的比例
# # V1 = math.pi/(3*((3)**0.5))
# V2 = (1-V3)-V1
# S3 = int(n3/m)
# n = int(S1 + S2 + S3)   # 一共分多少层
#
# CC = [0]
# linshicanshu = 0  # 临时参数
# shouxiang = 1 # 首项
# # print(S1,S2,V1,V2,n1)
# for i in range(1,S1+1):
#     linshicanshu = int((shouxiang + (shouxiang+(i-1)*(m-2)) )*i /2)
#     CC.append((V1/(n1/m))*linshicanshu)
# shouxiang = (1+(S2-1)*(m-2))
# MM = (1+1+(S2-1)*(m-2))*S2/2
# for i in range(1,S2+1):
#     linshicanshu = int((shouxiang + (shouxiang - (i - 1) * (m - 2))) * i / 2)
#     CC.append(CC[S1]+(V2 / MM) * linshicanshu)
# linshicanshu = (1-CC[S1+S2])/S3
# for i in range(S3):
#     CC.append(CC[-1] +  linshicanshu)


# n = 18  # 层数：子集有多少个  # 分三次选 前3分之一选1-6  1-3  1-2

# 自己选择
subset = []
Ssubset = np.array([1,0.5889359,	0.5818121,	0.5857817,1])
# print(N)
n=10
nn=5

for i in range(n):  # 每层选一个
    # print("选{}个点".format(i + 1))

    i_ceng = sorted_matrix[(i) * int(N / n):(i + 1) * int(N / n)]

    subset = i_ceng[:, 1:4]
    data1 = i_ceng


    pro_rank = np.argsort(i_ceng[:, m + 1])  # 按乘积排名，从小到大
    SSsubset = np.array([1, 0, 0, 0, 1])
    for j in range(nn):
        AAA = int(j / (nn) * (len(data1)))

        Ssubset = np.vstack((Ssubset, data1[pro_rank[AAA]]))
        SSsubset = np.vstack((SSsubset, data1[pro_rank[AAA]]))



    # SSSsubset = SSsubset[:,1:4]
    # print(SSSsubset)
    # x = SSSsubset[:, 0]
    # y = SSSsubset[:, 1]
    # z = SSSsubset[:, 2]
    #
    # # 创建一个新的三维图
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # 绘制三维点阵
    # ax.scatter(x, y, z, c='b', marker='o')
    #
    # # 添加轴标签
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis')
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_zlim(0, 1)
    # plt.title('3D Point Cloud from Matrix Columns')
    #
    # # 显示图表
    # plt.show(block=True)
    #

end_time = time.time() # 记录时间结束

# 1012 问题有重复的点被选择了
#     print(len(SSsubset))
#     print(SSsubset[:,1:4])

# print(Ssubset)
Ssubset = np.delete(Ssubset, 0, axis=0)
# print(Ssubset)
Ssubset = Ssubset[:,1:4]
x = Ssubset[:, 0]
y = Ssubset[:, 1]
z = Ssubset[:, 2]

# 创建一个新的三维图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制三维点阵
ax.scatter(x, y, z, c='b', marker='o')
ax.view_init(elev=9, azim=-42)
# 添加轴标签
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
plt.title('Subset of RBSS')
Ssubset = Ssubset.astype(float)
print(ea.indicator.HV(Ssubset))
print(ea.indicator.Spacing(Ssubset))
R = np.array([[1,1,1]])
print(ea.indicator.IGD(Ssubset,R))

# 显示图表
print("时间为：",end_time - start_time)
# plt.savefig('RBSS2DWFG3.png')
plt.show(block=True)
print(len(Ssubset))
# print(Ssubset)
# print(ea.indicator.HV(SSsubset))




#  每层 + m-1
#  选点
#1109   第四层出现问题，没有按照代码逻辑呈现

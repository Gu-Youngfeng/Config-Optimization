#!\usr\bin\python
#coding=utf-8

"""
This code is used to check my immediate and simple new idea.
"""

import matplotlib.pyplot as plt
import numpy as np

########################################################################################## 
# min actual rank in top-10
##########################################################################################

# data from csv
data_1=[1.28, 28.02, 34.799999999999997, 48.340000000000003, 14.1, 1.3400000000000001, 37.539999999999999, 53.039999999999999, 34.259999999999998, 21.199999999999999, 296.0, 116.7, 0.5, 0.22, 0.28000000000000003, 0.26000000000000001, 0.5, 0.40000000000000002, 4.8399999999999999, 28.760000000000002, 32.740000000000002, 3.3599999999999999, 4.2800000000000002]
x1 = range(len(data_1))

plt.subplot(131)
plt.scatter(x1, data_1, marker='o')
plt.xticks(x1, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'LLVM', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'spear', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements'), rotation=90)
plt.title("Data from csv file")
plt.ylabel("Rank Difference")

# data from result table
data_2 = []
x2 = range(len(data_2))

plt.subplot(132)
plt.scatter(x2, data_2, marker='o')
plt.xticks(x2, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'LLVM', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'spear', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements'), rotation=90)
plt.title("Data from my code")

# data from (Nair et al. 2017)
data_3 = [0.29999999999999999, 5.5499999999999998, 7.3499999999999996, 4.2999999999999998, 2.6000000000000001, 0.050000000000000003, 27.25, 7.4000000000000004, 15.800000000000001, 7.6500000000000004, 4.0, 52.549999999999997, 0.59999999999999998, 0.65000000000000002, 0.34999999999999998, 0.25, 0.65000000000000002, 0.34999999999999998, 1.2, 16.300000000000001, 4.0, 3.3500000000000001, 1.3]
x3 = range(len(data_3))

plt.subplot(133)
plt.scatter(x3, data_3, marker='o')
plt.xticks(x3, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'LLVM', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'spear', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements'), rotation=90)
plt.title("Data from Nair et al.")

plt.show()

########################################################################################## 
# nums of configurations predicted as top 1
##########################################################################################

# data = [5.2999999999999998, 35.280000000000001, 33.579999999999998, 49.640000000000001, 20.940000000000001, 5.4800000000000004, 46.0, 59.159999999999997, 39.68, 33.619999999999997, 708.58000000000004, 94.599999999999994, 3.98, 4.0, 4.46, 3.7400000000000002, 5.0, 4.46, 15.640000000000001, 35.359999999999999, 47.560000000000002, 4.2000000000000002, 22.140000000000001]
# x = range(len(data))

# plt.xlabel("project")
# plt.ylabel("same predicted top-1")

# plt.bar(x, data)
# plt.xticks(x, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'LLVM', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'spear', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements'), rotation=90)
# plt.show()

########################################################################################### 
# selected data distribution and the whole data distribution
###########################################################################################

# import pandas as pd
# import numpy as np
# import random as rd

# pdcontent = pd.read_csv("../data/X264_AllMeasurements.csv")

# performance = [attr for attr in pdcontent.columns if "$<" in attr]
# perfs_name = performance[-1]

# # [SORT BY PERFORMANCE]
# sortedcontent = pdcontent.sort_values(perfs_name)
# perfs = []
# for i in range(len(sortedcontent)):
# 	perfs.append(sortedcontent.iloc[i][perfs_name])
# print(perfs)

# # [SELECT RANDOMLY]
# pd_perfs=[]
# for i in range(len(pdcontent)):
# 	pd_perfs.append(pdcontent.iloc[i][perfs_name])
# rd.shuffle(pd_perfs)

# M = 100 # selected configuration
# selected_perfs=[]
# for i in range(M):
# 	selected_perfs.append(pd_perfs[i])
# print(selected_perfs)

# # [PERFORMANCE]
# config_num = len(perfs)
# x = range(config_num)
# plt.plot(x, perfs)
# plt.xlabel("configurations")
# plt.ylabel("performance")
# plt.show()

# # [DISTRIBUTION]
# plt.hist(perfs, 10, alpha=0.5) # whole distribution
# plt.hist(selected_perfs, 10) # whole distribution
# plt.show()



#!\usr\bin\python
#coding=utf-8

import matplotlib.pyplot as plt
import numpy as np

########################################################################################## 
# min actual rank in top-10
##########################################################################################

# x_x264 = [35.0, 1.0, 4.0, 1.0, 29.0, 2.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 20.0, 1.0, 1.0, 25.0, 1.0, 11.0, 37.0, 1.0, 15.0, 43.0, 83.0, 1.0, 22.0, 45.0, 1.0, 1.0, 1.0, 1.0, 11.0, 1.0, 1.0, 43.0, 1.0, 22.0, 1.0, 36.0, 30.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 20.0]
# x_lrzip = [3.0, 1.0, 2.0, 1.0, 21.0, 1.0, 1.0, 1.0, 17.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 3.0, 1.0, 1.0, 6.0, 8.0, 2.0, 1.0, 1.0, 9.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 21.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 4.0, 1.0, 1.0, 1.0]
# x_apache = [1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 21.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 9.0, 1.0, 1.0, 1.0]
# x_rs_6d_c3_obj2 = [30.0, 15.0, 1.0, 17.0, 1.0, 1.0, 20.0, 32.0, 1.0, 47.0, 1.0, 51.0, 16.0, 45.0, 104.0, 56.0, 1.0, 1.0, 1.0, 1.0, 6.0, 43.0, 23.0, 21.0, 1.0, 73.0, 55.0, 39.0, 168.0, 54.0, 39.0, 1.0, 39.0, 24.0, 19.0, 38.0, 1.0, 312.0, 1.0, 34.0, 32.0, 199.0, 1.0, 275.0, 1.0, 27.0, 28.0, 1.0, 1.0, 62.0]

# data = [x_x264, x_lrzip, x_apache, x_rs_6d_c3_obj2]

# plt.xlabel("project")
# plt.ylabel("min actual rank in top-10")

# plt.boxplot(data)
# plt.xticks(range(len(data)+1), ("","x264", "lrzip", "apache", "rs_6d_c3_obj2"))
# plt.show()

########################################################################################## 
# same predicted rank in top-10
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

import pandas as pd
import numpy as np
import random as rd

pdcontent = pd.read_csv("../data/X264_AllMeasurements.csv")

performance = [attr for attr in pdcontent.columns if "$<" in attr]
perfs_name = performance[-1]

# [SORT BY PERFORMANCE]
sortedcontent = pdcontent.sort_values(perfs_name)
perfs = []
for i in range(len(sortedcontent)):
	perfs.append(sortedcontent.iloc[i][perfs_name])
print(perfs)

# [SELECT RANDOMLY]
pd_perfs=[]
for i in range(len(pdcontent)):
	pd_perfs.append(pdcontent.iloc[i][perfs_name])
rd.shuffle(pd_perfs)

M = 100 # selected configuration
selected_perfs=[]
for i in range(M):
	selected_perfs.append(pd_perfs[i])
print(selected_perfs)

# [PERFORMANCE]
config_num = len(perfs)
x = range(config_num)
plt.plot(x, perfs)
plt.xlabel("configurations")
plt.ylabel("performance")
plt.show()

# [DISTRIBUTION]
plt.hist(perfs, 10, alpha=0.5) # whole distribution
plt.hist(selected_perfs, 10) # whole distribution
plt.show()



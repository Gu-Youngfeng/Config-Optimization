#!\usr\bin\python
#coding=utf-8
# Author: youngfeng
# Update: 07/16/2018

"""
This code is used to check my immediate and simple new idea.
"""

import matplotlib.pyplot as plt
import numpy as np

########################################################################################## 
# Figure 1: min actual rank in top-10
##########################################################################################

# # numeric projects
# projs1 = ['noc', 'rs-6d-c3-obj1', 'rs-6d-c3-obj2', 'snw', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4-obj1', 'wc-3d-c4-obj2', 'wc-5d-c5-obj1', 'wc-5d-c5-obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'wc-c1-3d-c1-obj1', 'wc-c1-3d-c1-obj2', 'wc-c3-3d-c1-obj1', 'wc-c3-3d-c1-obj2']
# # boolean projects
# projs2 = ['AJStats', 'Apache', 'BerkeleyC', 'BerkeleyJ', 'clasp', 'Dune', 'Hipacc', 'HSMGP_num', 'LLVM', 'lrzip', 'sac', 'spear', 'SQL', 'WGet', 'x264', 'XZ']

# data11 = [5.1399999999999997, 108.88, 141.38, 5.0800000000000001, 145.18000000000001, 97.540000000000006, 14.84, 4.1200000000000001, 12.66, 4.2599999999999998, 19.82, 5.04, 34.640000000000001, 25.0, 99.599999999999994, 26.18, 289.48000000000002, 112.95999999999999, 76.959999999999994, 96.459999999999994, 89.159999999999997, 79.599999999999994]
# data21 = [5746.1599999999999, 6.6399999999999997, 154.12, 11.699999999999999, 25.719999999999999, 126.02, 1212.76, 178.06, 74.700000000000003, 21.079999999999998, 54.719999999999999, 1010.08, 697.75999999999999, 24.800000000000001, 99.459999999999994, 88.560000000000002]

# data15 = [1.5800000000000001, 38.740000000000002, 45.200000000000003, 1.5800000000000001, 56.560000000000002, 28.739999999999998, 2.96, 0.62, 2.48, 1.0, 3.2999999999999998, 0.73999999999999999, 7.6799999999999997, 6.3600000000000003, 28.359999999999999, 9.0, 68.459999999999994, 31.920000000000002, 12.68, 28.379999999999999, 31.719999999999999, 27.760000000000002]
# data25 = [1795.0599999999999, 1.0600000000000001, 46.799999999999997, 2.8399999999999999, 6.6399999999999997, 37.460000000000001, 311.36000000000001, 58.100000000000001, 12.92, 3.2200000000000002, 11.56, 272.04000000000002, 284.16000000000003, 7.04, 28.800000000000001, 24.699999999999999]

# data110 = [0.76000000000000001, 25.640000000000001, 30.800000000000001, 0.41999999999999998, 25.859999999999999, 16.280000000000001, 1.74, 0.23999999999999999, 1.3200000000000001, 0.52000000000000002, 1.3, 0.29999999999999999, 4.04, 3.2599999999999998, 12.880000000000001, 4.6200000000000001, 25.52, 18.239999999999998, 7.7599999999999998, 15.619999999999999, 19.120000000000001, 12.300000000000001]
# data210 = [807.63999999999999, 0.29999999999999999, 25.280000000000001, 1.28, 4.1200000000000001, 19.5, 202.5, 25.140000000000001, 6.9800000000000004, 1.48, 7.7800000000000002, 156.12, 140.38, 3.3799999999999999, 18.760000000000002, 15.779999999999999]

# data120 = [0.76000000000000001, 25.640000000000001, 30.800000000000001, 0.41999999999999998, 25.859999999999999, 16.280000000000001, 1.74, 0.23999999999999999, 1.3200000000000001, 0.52000000000000002, 1.3, 0.29999999999999999, 4.04, 3.2599999999999998, 12.880000000000001, 4.6200000000000001, 25.52, 18.239999999999998, 7.7599999999999998, 15.619999999999999, 19.120000000000001, 12.300000000000001]
# data220 = [807.63999999999999, 0.29999999999999999, 25.280000000000001, 1.28, 4.1200000000000001, 19.5, 202.5, 25.140000000000001, 6.9800000000000004, 1.48, 7.7800000000000002, 156.12, 140.38, 3.3799999999999999, 18.760000000000002, 15.779999999999999]

# projs = projs1 + projs2

# data_1 = data11 + data21

# data_5 = data15 + data25

# data_10 = data110 + data210

# data_20 = data120 + data220

# # if you want to delete some project
# del_lst = [0, 3]
# offside = 0
# for i in del_lst:
# 	del projs[i - offside]
# 	del data_1[i - offside]
# 	del data_5[i - offside]
# 	del data_10[i - offside]
# 	del data_20[i - offside]
# 	offside += 1

# x1 = range(len(projs))

# plt.plot(x1, data_1, x1, data_5, x1, data_10, x1, data_20)
# plt.xticks(x1, tuple(projs), rotation=90)
# plt.title("minRank by selecting top-k configurations")
# # plt.yscale("log")
# plt.ylabel("Rank Difference")
# plt.legend(["k = 1","k = 5","k = 10","k = 20"])

# plt.show()


########################################################################################## 
# Figure 2: nums of configurations predicted as top 1
##########################################################################################

# # numeric projects
# projs1 = ['noc', 'rs-6d-c3-obj1', 'rs-6d-c3-obj2', 'snw', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4-obj1', 'wc-3d-c4-obj2', 'wc-5d-c5-obj1', 'wc-5d-c5-obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'wc-c1-3d-c1-obj1', 'wc-c1-3d-c1-obj2', 'wc-c3-3d-c1-obj1', 'wc-c3-3d-c1-obj2']
# # boolean projects
# projs2 = ['AJStats', 'Apache', 'BerkeleyC', 'BerkeleyJ', 'clasp', 'Dune', 'Hipacc', 'HSMGP_num', 'LLVM', 'lrzip', 'sac', 'spear', 'SQL', 'WGet', 'x264', 'XZ']


# data_1 = [5.0999999999999996, 66.780000000000001, 73.180000000000007, 4.0599999999999996, 51.640000000000001, 30.260000000000002, 5.6399999999999997, 4.0, 4.2999999999999998, 3.46, 5.04, 4.0599999999999996, 14.6, 13.119999999999999, 26.68, 20.34, 49.219999999999999, 41.960000000000001, 22.82, 24.539999999999999, 23.559999999999999, 23.02]


# data_2 = [763.41999999999996, 5.3799999999999999, 50.340000000000003, 3.8599999999999999, 10.779999999999999, 33.659999999999997, 299.98000000000002, 43.700000000000003, 24.280000000000001, 9.0800000000000001, 18.140000000000001, 431.24000000000001, 67.340000000000003, 4.4199999999999999, 42.359999999999999, 21.920000000000002]


# # if you want to delete some project
# del_lst = [0, 3]
# offside = 0
# for i in del_lst:
# 	del projs1[i - offside]
# 	del data_1[i - offside]
# 	offside += 1

# x1 = len(projs1)
# x2 = len(projs2)

# x = range(x1+x2)

# plt.ylabel("configurations (in log scale)")
# plt.bar(x[:x1], data_1,color="green",log=True)
# plt.bar(x[x1:], data_2, color="blue",log=True)

# plt.ylim(0, 1000)

# plt.xticks(x, tuple(projs1 + projs2), rotation=90)
# plt.title("configurations predicted as top 1 in each project")
# plt.legend(["numeric", "boolean"])

# plt.show()


###########################################################################################################
#
#  Nair's Faulty Sorting
#
######################

# # numeric projects
# projs1 = ['noc', 'rs-6d-c3-obj1', 'rs-6d-c3-obj2', 'snw', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4-obj1', 'wc-3d-c4-obj2', 'wc-5d-c5-obj1', 'wc-5d-c5-obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'wc-c1-3d-c1-obj1', 'wc-c1-3d-c1-obj2', 'wc-c3-3d-c1-obj1', 'wc-c3-3d-c1-obj2']
# # boolean projects
# projs2 = ['AJStats', 'Apache', 'BerkeleyC', 'BerkeleyJ', 'clasp', 'Dune', 'Hipacc', 'HSMGP_num', 'LLVM', 'lrzip', 'sac', 'spear', 'SQL', 'WGet', 'x264', 'XZ']

# projs = projs1 + projs2

# data_own = [0.76, 25.64, 30.8, 0.42, 16.28, 25.86, 1.74, 0.24, 1.32, 0.52, 1.3, 0.3, 4.04, 3.26, 12.88, 4.62, 25.52, 18.24, 7.76, 15.62, 19.12, 12.3, 807.64, 0.3, 25.28, 1.28, 4.12, 19.5, 202.5, 25.14, 6.98, 1.48, 7.78, 156.12, 140.38, 3.38, 18.76, 15.78]

# data_nair = [0.5, 12.66, 9.38, 0.34, 8.18, 11.04, 1.52, 0.2, 0.96, 0.4, 1.08, 0.14, 1.92, 2.1, 5.72, 2.24, 6.08, 6.84, 2.82, 7.96, 9.82, 3.8, 42.62, 0.18, 14.06, 0.8, 3.04, 9.26, 31.86, 8.38, 3.14, 1.04, 4.44, 17.02, 52.86, 2.66, 5.48, 9.72]


# # if you want to delete some project
# del_lst = [0, 3]
# offside = 0
# for i in del_lst:
# 	del projs[i - offside]
# 	del data_own[i - offside]
# 	del data_nair[i - offside]
# 	offside += 1

# x = range(len(projs))

# plt.ylabel("Rank Difference (in log scale)")
# plt.plot(x, data_own)
# plt.plot(x, data_nair)

# plt.yscale("log")
# plt.ylim(0, 1000)

# plt.xticks(x, tuple(projs), rotation=90)
# plt.title("Faulty Sort used by Nair et al.")
# plt.legend(["Shuffled sort", "Nair's sort"])

# plt.show()


########################################################################################### 
# Figure 3: selected data distribution and the whole data distribution
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



########################################################################################### 
# Figure 4: difference among 3 ranks
###########################################################################################

# data_min_meida = [4.1699999999999999, 18.350000000000001, 20.02, 33.369999999999997, 9.4700000000000006, 5.1299999999999999, 34.049999999999997, 23.760000000000002, 24.73, 15.31, 370.07999999999998, 96.799999999999997, 3.73, 1.73, 2.2000000000000002, 2.2000000000000002, 2.6000000000000001, 2.6800000000000002, 8.0899999999999999, 18.370000000000001, 22.899999999999999, 9.0700000000000003, 9.2799999999999994]
# data_media_media = [7.96, 108.87, 160.5, 119.87, 25.800000000000001, 7.4199999999999999, 185.19999999999999, 73.739999999999995, 63.539999999999999, 62.880000000000003, 535.22000000000003, 849.41999999999996, 5.0899999999999999, 2.3900000000000001, 3.1099999999999999, 3.1000000000000001, 3.2599999999999998, 3.6200000000000001, 13.31, 92.590000000000003, 63.920000000000002, 19.66, 19.309999999999999]
# data_rd = [1.4199999999999999, 10.859999999999999, 11.380000000000001, 6.5, 3.4199999999999999, 0.64000000000000001, 18.800000000000001, 3.5600000000000001, 17.039999999999999, 6.8200000000000003, 3.5600000000000001, 47.719999999999999, 0.59999999999999998, 0.20000000000000001, 0.46000000000000002, 0.29999999999999999, 0.59999999999999998, 0.28000000000000003, 2.0800000000000001, 6.54, 11.26, 2.3599999999999999, 2.1200000000000001]

# x = range(len(data_min_meida))

# plt.plot(x, data_min_meida, "b", x, data_media_media, "r", x, data_rd, "g")
# # plt.yscale('log')

# plt.ylabel("rank difference")
# plt.xticks(x, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'LLVM', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'spear', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements'), rotation=90)
# plt.title("rank difference mearsured by 3 methods")
# plt.legend(["RD1", "RD2", "RD3"])

# plt.show()



########################################################################################### 
# Figure 5: pearson correlation between (minR-mediaP),(mediaR-mediaP),rd and top-1
###########################################################################################

# from scipy import stats

# data_min_meida = [4.17,18.35,20.02,33.37,9.47,5.13,34.05,23.76,24.73,15.31,370.08,96.8,8.09,18.37,22.9,3.73,1.73,2.2,2.2,2.6,2.68,9.07,9.28]
# data_media_media = [7.96,108.87,160.5,119.87,25.8,7.42,185.2,73.74,63.54,62.88,535.22,949.42,13.31,92.59,63.92,5.09,2.39,3.11,3.1,3.26,3.62,19.66,19.31]
# data_rd = [1.42,10.86,11.38,6.5,3.42,0.64,18.8,3.56,17.04,6.82,3.56,47.72,2.08,6.54,11.26,0.6,0.2,0.46,0.3,0.6,0.28,2.36,2.12]

# data_top1 = [6.0199999999999996, 36.859999999999999, 42.479999999999997, 63.060000000000002, 18.420000000000002, 7.0999999999999996, 70.739999999999995, 54.159999999999997, 41.299999999999997, 37.060000000000002, 748.27999999999997, 103.0, 5.3799999999999999, 4.0999999999999996, 4.7999999999999998, 4.7199999999999998, 4.2000000000000002, 4.7199999999999998, 16.460000000000001, 43.740000000000002, 47.560000000000002, 4.8200000000000003, 21.719999999999999]

# p_min_media = stats.pearsonr(data_top1, data_min_meida)
# p_media_media = stats.pearsonr(data_top1, data_media_media)
# p_rd = stats.pearsonr(data_top1, data_rd)

# print(p_media_media, p_media_media, p_rd)

# plt.subplot(131)
# plt.scatter(data_top1, data_min_meida)
# plt.ylim(0,120)
# plt.xlim(0,150)
# plt.xlabel("number of configurations predicted as top 1")
# plt.ylabel("RD1")
# plt.title("pearson correlation (p = 0.54)")

# plt.subplot(132)
# plt.scatter(data_top1, data_media_media)
# plt.ylim(0,210)
# plt.xlim(0,160)
# plt.xlabel("number of configurations predicted as top 1")
# plt.ylabel("RD2")
# plt.title("pearson correlation (p = 0.54)")

# plt.subplot(133)
# plt.scatter(data_top1, data_rd)
# plt.ylim(0,50)
# plt.xlim(0,120)
# plt.xlabel("number of configurations predicted as top 1")
# plt.ylabel("RD3")
# plt.title("pearson correlation (p = 0.06)")
# plt.show()




########################################################################################### 
# Figure 6: statistics of 23 projects
###########################################################################################

# import pandas as pd
# import os

# projects = ["../data/" + file for file in os.listdir("../data") if ".csv" in file]
# # print(projects)

# proj_config = []
# proj_fea = []
# for proj in projects:
# 	pdcontent = pd.read_csv(proj)
# 	proj_config.append(len(pdcontent))
# 	proj_fea.append(len(pdcontent.columns)-1)
# 	# print("[project]:", proj, "[configuration]:", len(pdcontent))
# print("[configurations]:", proj_config)
# print("[option    size]:",proj_fea)

# x = range(len(proj_config))

# # subplot 1. configuration contained in each project
# num_index = [6,7,8,9,12,13,14,15,16,17,18,19,20]
# num_proj_config = [proj_config[i] for i in num_index]

# hyb_index = [2,3]
# hyb_proj_config = [proj_config[i] for i in hyb_index]

# plt.bar(x, proj_config, color="#239a3b")
# plt.bar(num_index, num_proj_config, color="#0366d6")
# plt.bar(hyb_index, hyb_proj_config, color="red")

# plt.xticks(x, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'LLVM', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'spear', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements'), rotation=90)
# plt.ylabel("configurations")
# # plt.ylim(10, 20000)
# plt.title("configurations in each project")
# plt.legend(["boolean project","numeric projects", "hybrid projects"])

# plt.show()

# # subplot 2. option size in each project
# num_index = [6,7,8,9,12,13,14,15,16,17,18,19,20]
# num_proj_fea = [proj_fea[i] for i in num_index]

# hyb_index = [2,3]
# hyb_proj_fea = [proj_fea[i] for i in hyb_index]

# plt.bar(x, proj_fea, color="#239a3b")
# plt.bar(num_index, num_proj_fea, color="#0366d6")
# plt.bar(hyb_index, hyb_proj_fea, color="red")

# plt.xticks(x, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'LLVM', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'spear', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements'), rotation=90)
# plt.ylabel("options")
# plt.title("options in each project")
# plt.legend(["boolean project","numeric projects", "hybrid projects"])

# plt.show()




########################################################################################### 
# Figure 7: the optimal configuration in 23 projects
###########################################################################################

# import pandas as pd
# import os

# projects = ["../data/" + file for file in os.listdir("../data") if ".csv" in file]
# print(projects)

# top_1_lst = []
# ave_count_percent_lst = []

# for proj in projects:
# 	pdcontent = pd.read_csv(proj)
	
# 	attr = pdcontent.columns
# 	feas = attr[:-1]
# 	perf = attr[-1]
# 	sortedcontent = pdcontent.sort_values(perf)
# 	# print(features)

# 	best_perf = sortedcontent.iloc[0][perf]

# 	# print(best_perf)

# 	best_config = []

# 	for i in range(len(sortedcontent)):
# 		if sortedcontent.iloc[i][perf] == best_perf:
# 			best_config.append(sortedcontent.iloc[i])
# 		else:
# 			break

# 	top_1 = len(best_config)
# 	top_1_lst.append(top_1)
# 	print(proj, ":", top_1)

# 	count_percent_lst = []
# 	for config in best_config:
# 		print(">>", config[feas].tolist(), ":", config[perf])
# 		count = 0
# 		for i in config[feas].tolist():
# 			if i == 0.0:
# 				count += 1

# 		count_percent = count/len(config[feas].tolist())
# 		print("[zero options]:", count, "[zero percents]:", count_percent)
# 		count_percent_lst.append(count_percent)

# 	print("------------")
# 	ave_count_percent_lst.append(np.mean(count_percent_lst))

# print(top_1_lst)
# print(ave_count_percent_lst)

###################################################################

# x= range(len(ave_count_percent_lst))
# plt.bar(x, ave_count_percent_lst)

# plt.xticks(x, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'LLVM', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'spear', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements'), rotation=90)
# plt.ylabel("zero ratio")
# # plt.ylim(10, 20000)
# plt.title("zero ratios in optimal configurations in each project")

# plt.show()

###################################################################

# one_percent = [1/6, 1/6, 0, 1/3, 2/3, 1/3, 2/3, 1/3, 2/3, 1/3, 2/3, 1/3, 1/2]
# x= range(len(one_percent))
# plt.bar(x, one_percent)

# plt.xticks(x, ('rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2'), rotation=90)
# plt.ylabel("ratio")
# plt.ylim(0.0, 1.0)
# plt.title("lowest ratios in optimal configurations in numeric projects")

# plt.show()

###################################################################

# zero_percent = [0.5, 0.66, 0.73, 0.84, 0.5, 0.28, 0.625, 0.5]
# x= range(len(zero_percent))
# plt.bar(x, zero_percent)

# plt.xticks(x, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'LLVM', 'lrzip', 'spear', 'SQL_AllMeasurements', 'WGet', 'X264_AllMeasurements'), rotation=90)
# plt.ylabel("ratio")
# plt.ylim(0.0, 1.0)
# plt.title("zero ratios in optimal configurations in boolean projects")

# plt.show()

###################################################################

# zero_lowest_percent = [7/11, 11/14]
# zero_percent = [6/8, 9/11]
# lowest_persent = [1/3, 2/3] 

# x= [1, 1.5]
# x_2 = [i+0.1 for i in x]
# x_3 = [i+0.2 for i in x]

# plt.bar(x, zero_percent, width=0.1)
# plt.bar(x_2, lowest_persent, width=0.1)
# plt.bar(x_3, zero_lowest_percent, width=0.1)

# plt.xticks(x, ('Dune', 'HSMGP_num'), rotation=90)
# plt.ylabel("ratio")
# plt.ylim(0.0, 1.0)
# plt.title("zero/lowest ratios in optimal configurations in hybrid projects")
# plt.legend(["zero ratio", "lowest ratio", "hybrid ratio"])

# plt.show()





########################################################################################### 
# Table 1: baisc info of 23 projects
###########################################################################################

# import pandas as pd
# import os

# projects = ["../data/" + file for file in os.listdir("../data") if ".csv" in file]
# # print(projects)

# proj_config = []
# proj_features = []
# for proj in projects:
# 	pdcontent = pd.read_csv(proj)
# 	proj_config.append(len(pdcontent))
# 	proj_features.append(len(pdcontent.columns)-1)
# 	# print("[project]:", proj, "[configuration]:", len(pdcontent))
# print("[configurations]:", proj_config)
# print("[option    size]:",proj_features)

# x = range(len(proj_config))

# # subplot 1. configuration contained in each project
# num_index = [2,3,6,7,8,9,12,13,14,15,16,17,18,19,20]
# from_index = [1 ,1 ,1 ,1 ,3 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]

# projs = ['Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'LLVM', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'spear', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements']

# print("| Project | Features | Configurations | Feature Mode | Dataset |")
# print("| :-- | :-- | :-- | :-- | :-- | :-- |")
# for i in x:
# 	isNumeric = "boolean"
# 	if i in num_index:
# 		isNumeric = "numeric"
# 	print("|", projs[i], "|", proj_features[i], "|", proj_config[i], "|", isNumeric, "|", from_index[i], "|")


########################################################################################### 
# Figure 8: variance between actual and optimal performance of configurations predicted as top 1
###########################################################################################

# varss = [7175.4160986576971, 0.042479849261876354, 1878587.9569974444, 2603.2947950439575, 77.978308574511843, 1392813719.5719535, 351421259.23887461, 48713.249340959061, 6956544.2086909087, 12175.035780795044, 0.00084401865517270329, 0.558313104399906, 13879284.245847931, 13378003.765060468, 3748715.3275375883, 13.644426407392544, 5737296.4723959537, 12.525749821529269, 2.0861860116409838, 9322485.331158841, 73.71870187838941, 8859564.3537099101, 265.10225328156366]
# x = range(len(varss))

# plt.bar(x, varss, log=True)
# plt.xticks(x, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'LLVM', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'spear', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements'), rotation=90)
# plt.ylim(0, 10000000000)

# plt.ylabel("variance (in log scale)")
# plt.title("variance of actual performances of configurations predicted as top 1")

# plt.show()



########################################################################################### 
# BUG: Find bug in find_lowesr_rank(train, test)
###########################################################################################
# import random as rd
# predicted = [350,150,150,700,600,150,150,150,150,150,150,150]

# predicted_id = [[(i+1), p] for i, p in enumerate(predicted)]
# # rd.shuffle(predicted_id)
# predicted_sorted = sorted(predicted_id, key=lambda x: x[-1])

# print(predicted_sorted)


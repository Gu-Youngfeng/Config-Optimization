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

# # data from csv
# data_1=[1.4199999999999999, 10.859999999999999, 11.380000000000001, 6.5, 3.4199999999999999, 0.64000000000000001, 18.800000000000001, 3.5600000000000001, 17.039999999999999, 6.8200000000000003, 3.5600000000000001, 47.719999999999999, 0.59999999999999998, 0.20000000000000001, 0.46000000000000002, 0.29999999999999999, 0.59999999999999998, 0.28000000000000003, 2.0800000000000001, 6.54, 11.26, 2.3599999999999999, 2.1200000000000001]
# x1 = range(len(data_1))

# plt.subplot(131)
# plt.scatter(x1, data_1, marker='o')
# plt.xticks(x1, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'LLVM', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'spear', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements'), rotation=90)
# plt.title("Data from Yaoyao")
# plt.ylabel("Rank Difference")
# plt.ylim(-5, 60)

# # data from my code
# data_2 = [0.20000000000000001, 4.0, 2.0, 2.1499999999999999, 0.90000000000000002, 0.34999999999999998, 6.0999999999999996, 2.8999999999999999, 3.9500000000000002, 1.8500000000000001, 2.5499999999999998, 84.549999999999997, 0.29999999999999999, 0.20000000000000001, 0.0, 0.25, 0.0, 0.25, 0.25, 2.2000000000000002, 4.2000000000000002, 1.6000000000000001, 1.55]
# x2 = range(len(data_2))

# plt.subplot(132)
# plt.scatter(x2, data_2, marker='o')
# plt.xticks(x2, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'LLVM', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'spear', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements'), rotation=90)
# plt.title("Data from Yongfeng")
# plt.ylim(-5, 60)

# # data from (Nair et al. 2017)
# data_3 = [0.29999999999999999, 5.5499999999999998, 7.3499999999999996, 4.2999999999999998, 2.6000000000000001, 0.050000000000000003, 27.25, 7.4000000000000004, 15.800000000000001, 7.6500000000000004, 4.0, 52.549999999999997, 0.59999999999999998, 0.65000000000000002, 0.34999999999999998, 0.25, 0.65000000000000002, 0.34999999999999998, 1.2, 16.300000000000001, 4.0, 3.3500000000000001, 1.3]
# x3 = range(len(data_3))

# plt.subplot(133)
# plt.scatter(x3, data_3, marker='o')
# plt.xticks(x3, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'LLVM', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'spear', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements'), rotation=90)
# plt.title("Data from Rank-based")
# plt.ylim(-5, 60)

# plt.show()




########################################################################################## 
# Figure 2: nums of configurations predicted as top 1
##########################################################################################

# data = [6.0199999999999996, 36.859999999999999, 42.479999999999997, 63.060000000000002, 18.420000000000002, 7.0999999999999996, 70.739999999999995, 54.159999999999997, 41.299999999999997, 37.060000000000002, 748.27999999999997, 103.0, 5.3799999999999999, 4.0999999999999996, 4.7999999999999998, 4.7199999999999998, 4.2000000000000002, 4.7199999999999998, 16.460000000000001, 43.740000000000002, 47.560000000000002, 4.8200000000000003, 21.719999999999999]
# x = range(len(data))

# index = [0,7,9,21]
# data_t = [data[i] for i in index]

# index_2 = [10]
# data_t_2 = [data[i] for i in index_2]

# plt.ylabel("configurations")
# plt.bar(x, data,color="green",log=True)
# plt.bar(index, data_t, color="blue",log=True)
# plt.bar(index_2, data_t_2, color="red", log=True)
# plt.ylim(0, 1000)

# plt.xticks(x, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'LLVM', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'spear', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements'), rotation=90)
# plt.title("configurations predicted as top 1 in each project")
# plt.legend(["1 optimal config", "2 optimal configs", "1685 optimal configs"])

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

# data_min_meida = [4.17,18.35,20.02,33.37,9.47,5.13,34.05,23.76,24.73,15.31,370.08,96.8,8.09,18.37,22.9,3.73,1.73,2.2,2.2,2.6,2.68,9.07,9.28]
# data_media_media = [7.96,108.87,160.5,119.87,25.8,7.42,185.2,73.74,63.54,62.88,535.22,949.42,13.31,92.59,63.92,5.09,2.39,3.11,3.1,3.26,3.62,19.66,19.31]
# data_rd = [1.42,10.86,11.38,6.5,3.42,0.64,18.8,3.56,17.04,6.82,3.56,47.72,2.08,6.54,11.26,0.6,0.2,0.46,0.3,0.6,0.28,2.36,2.12]

# x = range(len(data_min_meida))

# plt.plot(x, data_min_meida, "b", x, data_media_media, "r", x, data_rd, "g")
# # plt.yscale('log')

# plt.ylabel("rank difference")
# plt.xticks(x, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'LLVM', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'spear', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements'), rotation=90)
# plt.title("Rank difference mearsured by 3 methods")
# plt.legend(["MinR-MediaP", "MediaR-MediaP", "MinR (Nair 2017)"])

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
# plt.ylabel("(minR - mediaP)")
# plt.title("pearson correlation (p = 0.54)")

# plt.subplot(132)
# plt.scatter(data_top1, data_media_media)
# plt.ylim(0,210)
# plt.xlim(0,160)
# plt.xlabel("number of configurations predicted as top 1")
# plt.ylabel("(mediaR - mediaP)")
# plt.title("pearson correlation (p = 0.54)")

# plt.subplot(133)
# plt.scatter(data_top1, data_rd)
# plt.ylim(0,50)
# plt.xlim(0,120)
# plt.xlabel("number of configurations predicted as top 1")
# plt.ylabel("rank difference")
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
# proj_features = []
# for proj in projects:
# 	pdcontent = pd.read_csv(proj)
# 	proj_config.append(len(pdcontent))
# 	proj_features.append(len(pdcontent.columns))
# 	# print("[project]:", proj, "[configuration]:", len(pdcontent))
# print("[configurations]:", proj_config)
# print("[option    size]:",proj_features)

# x = range(len(proj_config))

# # subplot 1. configuration contained in each project
# num_index = [2,3,6,7,8,9,12,13,14,15,16,17,18,19,20]
# num_proj_config = [proj_config[i] for i in num_index]

# plt.bar(x, proj_config, color="#239a3b")
# plt.bar(num_index, num_proj_config, color="#0366d6")

# plt.xticks(x, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'LLVM', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'spear', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements'), rotation=90)
# plt.ylabel("configuration size")
# # plt.ylim(10, 20000)
# plt.title("configuration size in each project")
# plt.legend(["boolean options","numeric options"])

# plt.show()

# # subplot 2. option size in each project
# plt.bar(x, proj_features)
# plt.xticks(x, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'LLVM', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'spear', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements'), rotation=90)
# plt.ylabel("option size")
# plt.title("option size in each project")

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

# x= range(len(ave_count_percent_lst))
# plt.bar(x, ave_count_percent_lst)

# plt.xticks(x, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'LLVM', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'spear', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements'), rotation=90)
# plt.ylabel("zero option ratio")
# # plt.ylim(10, 20000)
# plt.title("zero option ratios in optimal configurations in each project")

# plt.show()


one_percent = [1/6, 1/6, 0, 1/3, 2/3, 1/3, 2/3, 1/3, 2/3, 1/3, 2/3, 1/3, 1/2]
x= range(len(one_percent))
plt.bar(x, one_percent)

plt.xticks(x, ('rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2'), rotation=90)
plt.ylabel("one option ratio")
# plt.ylim(10, 20000)
plt.title("one option ratios in optimal configurations in 12 projects")

plt.show()





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
# 	proj_features.append(len(pdcontent.columns))
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


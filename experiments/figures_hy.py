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

# plt.ylabel("configurations (in log scale)")
# plt.bar(x, data,color="green",log=True)
# plt.bar(index, data_t, color="blue",log=True)
# plt.bar(index_2, data_t_2, color="red", log=True)
# plt.ylim(0, 1000)

# plt.xticks(x, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'LLVM', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'spear', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements'), rotation=90)
# plt.title("configurations predicted as top 1 in each project")
# plt.legend(["1 optimal", "2 optimals", "1685 optimals"])

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

rd = [0.050000000000000003, 18.600000000000001, 13.9, 7.0499999999999998, 0.94999999999999996, 15.449999999999999, 9.1999999999999993, 9.9499999999999993, 6.25, 56.25, 0.29999999999999999, 0.40000000000000002, 0.0, 0.5, 0.59999999999999998, 0.20000000000000001, 0.40000000000000002, 8.5999999999999996, 12.5, 2.0499999999999998, 3.7000000000000002]
x =range(len(rd))

for i in x:
	if rd[i]>=8:
		plt.scatter(i, rd[i], color="red")
	else:
		plt.scatter(i, rd[i], color="blue")

plt.xticks(x, ('Apache_AllMeasurements', 'BDBC_AllMeasurements', 'Dune', 'HSMGP_num', 'lrzip', 'rs-6d-c3_obj1', 'rs-6d-c3_obj2', 'sol-6d-c2-obj1', 'sol-6d-c2-obj2', 'SQL_AllMeasurements', 'wc+rs-3d-c4-obj1', 'wc+rs-3d-c4-obj2', 'wc+sol-3d-c4-obj1', 'wc+sol-3d-c4-obj2', 'wc+wc-3d-c4-obj1', 'wc+wc-3d-c4-obj2', 'wc-3d-c4_obj2', 'wc-6d-c1-obj1', 'wc-6d-c1-obj2', 'WGet', 'X264_AllMeasurements'), rotation=90)
plt.ylabel("Rank Difference (RD)")
plt.title("Rank-based")
plt.legend(["RD < 8", "RD > 8"])
plt.show()
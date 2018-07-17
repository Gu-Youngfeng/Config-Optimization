#!\usr\bin\python
# coding=utf-8
# Author: youngfeng
# Update: 07/17/2018

"""
This code is used to find the lowest rank in validation pool (top-10) by using flash method
"""

import sys
sys.path.append("../")
import os
from Flash import find_lowest_rank
from Flash import predict_by_flash
from Flash import split_data_by_fraction
import numpy as np

if __name__ == "__main__":

	projs = ["../data/"+name for name in os.listdir("../data") if ".csv" in name]

	ave_rank_lst = []

	for proj in projs:

		print(proj)

		ave_top_10_act_rank = []

		for round in range(50):

			# print(">> Using FLASH Method to Predict the Validation Pool\n")

			# select 80% data
			dataset = split_data_by_fraction(proj, 0.8)
			# print("### initialzation")
			# for i in dataset:
			# 	print(str(i.index), ",", end="")
			# print("\n-------------")
			data = predict_by_flash(dataset)

			# print("### finally split")
			train_set = data[0]
			uneval_set = data[1]
			# for i in train_set:
			# 	print(str(i.index), ",", end="")
			# print("\n-------------")
			# for i in uneval_set:
			# 	print(str(i.index), ",", end="")
			# print("\n-------------")

			lowest_rank = find_lowest_rank(train_set, uneval_set)

			ave_top_10_act_rank.append(lowest_rank)

		print("[mini rank]: ", ave_top_10_act_rank)
		minest_rank = np.mean(ave_top_10_act_rank)
		print("[mean rank]: ", minest_rank, "\n")

		ave_rank_lst.append(minest_rank)

	print("-------------------------------")
	print(ave_rank_lst)

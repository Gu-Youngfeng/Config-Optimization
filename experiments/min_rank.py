#!\usr\bin\python
# coding=utf-8
# Author: youngfeng
# Update: 07/15/2018

"""
This code is used to find the lowest rank in validation pool (top-10) by using rank based method
"""

import sys
sys.path.append("../")
import os
from RankBased import find_lowest_rank
from RankBased import predict_by_rank_based
from RankBased import predict_by_cart
from RankBased import split_data_by_fraction
import numpy as np

if __name__ == "__main__":

	print(">> Using Rank-based Method to Predict the Validation Pool\n")

	projects = ["../data/"+name for name in os.listdir("../data") if ".csv" in name]
	# projects = ["../data/Apache_AllMeasurements.csv"]
	print(projects)

	ave_rank_lst = []

	for proj in projects:
		# for each project
		print(proj)
		ave_top_10_act_rank = []
		ave_train_set = []
		for round in range(20):
			# data split
			split_data = split_data_by_fraction(proj, 0.4, round)
			train_pool = split_data[0]
			test_pool = split_data[1]
			validation_pool = split_data[2]

			# apply rank-based method on proj 
			# print("### Testing on Test Pool: ")
			train_set = predict_by_rank_based(train_pool, test_pool)
			ave_train_set.append(len(train_set))

			# print("\n--------------------")

			# evaluate on validation pool
			rd = predict_by_cart(train_set, validation_pool)
			# print("### Evaulation on Validation Pool: ", rd)

			lowest_rank = find_lowest_rank(train_set, validation_pool)

			ave_top_10_act_rank.append(lowest_rank)

			# DATASETS = []

		print("[mini rank]: ", ave_top_10_act_rank)
		minest_rank = np.mean(ave_top_10_act_rank)
		print("[mean rank]: ", minest_rank, "\n")

		# print("[train set]: ", ave_train_set, np.mean(ave_train_set))
		ave_rank_lst.append(minest_rank)

	print(ave_rank_lst)


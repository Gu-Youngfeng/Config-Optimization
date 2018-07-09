#!\usr\bin\python
# coding=utf-8
# Author: youngfeng
# Update: 07/09/2018

import sys
sys.path.append("..")
import os
from RankBased import predict_by_rank_based
from RankBased import predict_by_cart
from RankBased import split_data_by_fraction
import numpy as np


if __name__ == "__main__":

	projects = ["../data/"+name for name in os.listdir("../data/") if ".csv" in name]
	print(projects)

	for proj in projects[:]:
		# for each project
		print(proj)
		ave_top_10_act_rank = []

		for round in range(20):
			# data split
			split_data = split_data_by_fraction(proj, 0.4, round)
			train_pool = split_data[0]
			test_pool = split_data[1]
			validation_pool = split_data[2]

			# apply rank-based method on proj 
			# print("### Testing on Test Pool: ")
			train_set = predict_by_rank_based(train_pool, test_pool)

			# print("\n--------------------")

			# evaluate on validation pool
			rd = predict_by_cart(train_set, validation_pool)
			# print("### Evaulation on Validation Pool: ", rd)

			sorted_validation_pool = sorted(validation_pool, key=lambda x : x.rank)

			# sort the validation pool by predicted_perf
			# for config in sorted_validation_pool:
			# 	print(config.index, ",", config.perfs, ",", config.predicted, ",", config.rank)

			# find the min rank in top-10
			top_10_act_rank = []
			for config in sorted_validation_pool[:10]:
				top_10_act_rank.append(config.index)

			# print(np.min(top_10_act_rank))
			print(np.min(top_10_act_rank), " ", end="")

			ave_top_10_act_rank.append(np.min(top_10_act_rank))

			# DATASETS = []

		print("\n[AVE]: ", np.mean(ave_top_10_act_rank), "\n")


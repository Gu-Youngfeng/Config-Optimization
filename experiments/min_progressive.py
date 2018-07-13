#!\usr\bin\python
# coding=utf-8
# Author: youngfeng
# Update: 07/13/2018

"""
This code is used to find the lowest rank in validation pool (top-10) by using progressive method
"""

import sys
sys.path.append("..")
import os
from Progressive import find_lowest_rank
from Progressive import predict_by_progressive
from Progressive import predict_by_cart
from Progressive import split_data_by_fraction
import numpy as np

if __name__ == "__main__":

	print(">> Using Progressive Method to Predict the Validation Pool\n")

	projects = ["../data/"+name for name in os.listdir("../data/") if ".csv" in name]
	# projects = ["../data/Apache_AllMeasurements.csv"]
	print(projects)

	for proj in projects[1:2]:
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
			train_set = predict_by_progressive(train_pool, test_pool)

			# evaluate on validation pool
			mmre = predict_by_cart(train_set, validation_pool)
			# print("[mmre]: ", (1-mmre))

			lowest_rank = find_lowest_rank(train_set, validation_pool)
			ave_top_10_act_rank.append(lowest_rank)

		print("[mini rank]: ", ave_top_10_act_rank)
		print("[mean rank]: ", np.mean(ave_top_10_act_rank), "\n")

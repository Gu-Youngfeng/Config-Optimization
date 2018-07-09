#!\usr\bin\python
#coding=utf-8
# Author: youngfeng
# Update: 07/09/2018

"""
Rank-based method, proposed by Nair et al. (fse '17), is a configuration optimazation approach.
It uses relation-based measure (rank difference) instead of residual-based measure (mmre) to train 
predictive model itertively.
The details of Rank-based are introduced in paper "Using bad Learners to Find Good Configurations".
"""

import pandas as pd
import random as rd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
# for figures
import matplotlib.pyplot as plt 

# global variable COLLECTOR, to save points in learning curve 
COLLECTOR = []
# global variable DATASETS, to save train_pool, test_pool, validation_pool
DATASETS = []

class config_node:
	"""
	for each configuration, we create a config_node object to save its informations
	index    : actual rank
	features : feature list
	perfs    : actual performance
	rank     : predicted rank
	"""
	def __init__(self, index, features, perfs, predicted, rank):
		self.index = index
		self.features = features
		self.perfs = perfs
		self.predicted = predicted
		self.rank = rank


def predict_by_cart(train_set, test_set):

	train_fea_vector = [new_sample.features for new_sample in train_set]
	train_pef_vector = [new_sample.perfs[-1] for new_sample in train_set]

	test_fea_vector = [new_sample.features for new_sample in test_set]
	test_pef_vector = [new_sample.perfs[-1] for new_sample in test_set]

	##################### train model based on train set 
	##################### setting of minsplit and minbucket
	# S = len(train_set)
	# if S <= 100:
	# 	minbucket = np.floor((S/10)+(1/2))
	# 	minsplit = 2*minbucket
	# else:
	# 	minsplit = np.floor((S/10)+(1/2))
	# 	minbucket = np.floor(minsplit/2)

	# if minbucket < 2:
	# 	minbucket = 2
	# if minsplit < 4:
	# 	minsplit = 4

	# minbucket = int(minbucket) # cart cannot set a float minbucket or minsplit 
	# minsplit = int(minsplit)

	# cart_model = DecisionTreeRegressor( min_samples_split = minsplit,
	# 									min_samples_leaf = minbucket)
	#####################

	cart_model = DecisionTreeRegressor()
	cart_model.fit(train_fea_vector, train_pef_vector)
	test_pef_predicted = cart_model.predict(test_fea_vector)

	# predicted performance to config_node.predicted
	for (config, predicted_perf) in zip(test_set, test_pef_predicted):
		config.predicted[-1] = predicted_perf

	predicted_id = [[i,p] for i, p in enumerate(test_pef_predicted)]
	perdicted_sorted = sorted(predicted_id, key=lambda x: x[-1])
	predicted_rank_sorted = [[p[0], p[1], i] for i,p in enumerate(perdicted_sorted)] 
	#p[0] actual rank, p[1] predicted performance, i predicted rank

	for i in range(len(predicted_rank_sorted)):
		test_set[predicted_rank_sorted[i][0]].rank = predicted_rank_sorted[i][2]

	# predicted rank to config_node.rank

	rd_lst = []

	for config in test_set:
		rd = abs(config.rank - config.index)
		rd_lst.append(rd)

	return np.mean(rd_lst)


def predict_by_rank_based(csv_file, fraction, seed):
	# step1: read from csv file
	pdcontent = pd.read_csv(csv_file) 
	attr_list = pdcontent.columns # all feature list

	# step2: split attribute - method 1
	features = [i for i in attr_list if "$<" not in i]
	perfs = [i for i in attr_list if "$<" in i]
	sortedcontent = pdcontent.sort_values(perfs[-1]) # from small to big

	# step3: collect configuration
	configs = list()
	for c in range(len(pdcontent)):
		configs.append(config_node(c, # actual rank
									sortedcontent.iloc[c][features].tolist(), # feature list
									sortedcontent.iloc[c][perfs].tolist(), # performance list
									sortedcontent.iloc[c][perfs].tolist(), # predicted performance list
									c # predicted rank
			))

	# for config in configs:
	# 	print(config.index, "-", config.perfs, "-", config.predicted, "-", config.rank)

	# step4: data split
	# fraction = 0.4 # split fraction 
	rd.seed(seed) # random seed
	rd.shuffle(configs) # shuffle the configs
	indexes = range(len(configs))
	train_index = indexes[:int(fraction*len(configs))]
	test_index = indexes[int((fraction)*len(configs)): int((fraction+0.2)*len(configs))]
	validation_index = indexes[int((fraction+0.2)*len(configs)):]

	train_pool = [configs[i] for i in train_index]
	test_pool = [configs[i] for i in test_index]
	validation_pool = [configs[i] for i in validation_index]

	# DATASETS = [train_pool, test_pool, validation_pool]
	DATASETS.append(train_pool)
	DATASETS.append(test_pool)
	DATASETS.append(validation_pool)

	# for config in train_pool:
	# 	print(config.index, " ", end="")
	# print("-------------")
	# for config in test_pool:
	# 	print(config.index, " ", end="")
	# print("-------------")
	# for config in validation_pool:
	# 	print(config.index, " ", end="")

	# step5: initilize train set
	train_set = train_pool[:10]
	count = 10
	lives = 4
	last_rd = -1
	collector = []

	while lives>0 and count<len(train_pool):
		train_set.append(train_pool[count])
		count = count + 1

		current_rd = predict_by_cart(train_set, test_pool)

		print("[train]: ",count,", [rank difference]: ", current_rd)

		if current_rd >= last_rd:
			lives = lives - 1

		last_rd = current_rd 

	return train_set

if __name__ == "__main__":

	# apply rank-based method on proj 
	print("### Testing on Test Pool: ")
	train_set = predict_by_rank_based("data/Apache_AllMeasurements.csv", 0.4, 0)

	print("\n--------------------")

	# evaluate on validation pool
	validation_pool = DATASETS[2]
	rd = predict_by_cart(train_set, validation_pool)
	print("### Evaulation on Validation Pool: ", rd)

	sorted_validation_pool = sorted(validation_pool, key=lambda x : x.rank)

	# sort the validation pool by predicted_perf
	for config in sorted_validation_pool:
		print(config.index, ",", config.perfs, ",", config.predicted, ",", config.rank)

	# find the min rank in top-10
	top_10_act_rank = []
	for config in sorted_validation_pool[:10]:
		top_10_act_rank.append(config.index)

	print(np.min(top_10_act_rank))
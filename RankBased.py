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

import sys
import pandas as pd
import random as rd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
# for figures
import matplotlib.pyplot as plt 

# global variable COLLECTOR, to save points in learning curve 
COLLECTOR = []

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

def find_lowest_rank(train_set, test_set):

	sorted_test = sorted(test_set, key=lambda x: x.perfs[-1])
    
    # train data
	train_features = [t.features for t in train_set]
	train_perfs = [t.perfs[-1] for t in train_set]
    
    # test data
	test_perfs = [t.features for t in sorted_test]

	cart_model = DecisionTreeRegressor()
	cart_model.fit(train_features, train_perfs)
	predicted = cart_model.predict(test_perfs)

	predicted_id = [[i, p] for i, p in enumerate(predicted)]
    # i-> actual rank, p -> predicted value
	predicted_sorted = sorted(predicted_id, key=lambda x: x[-1])
    # print(predicted_sorted)
    # assigning predicted ranks
	predicted_rank_sorted = [[p[0], p[-1], i] for i,p in enumerate(predicted_sorted)]
    # p[0] -> actual rank, p[-1] -> perdicted value, i -> predicted rank
	select_few = predicted_rank_sorted[:10]

	# print the predcited top-10 configuration 
	# for sf in select_few:
	# 	print("actual rank:", sf[0], " actual value:", sorted_test[sf[0]].perfs[-1], " predicted value:", sf[1], " predicted rank:", sf[2])
	# print("-------------")

	return np.min([sf[0] for sf in select_few])


def predict_by_cart(train_set, test_set):

	test_set = sorted(test_set, key=lambda x:x.perfs[-1])

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

	# print("[min samples split]: ", minsplit)
	# print("[min samples leaf] : ", minbucket)

	# cart_model = DecisionTreeRegressor( min_samples_split = minsplit,
	# 									min_samples_leaf = minbucket,
	# 									max_depth = 30)

	cart_model = DecisionTreeRegressor() 
	cart_model.fit(train_fea_vector, train_pef_vector)
	predicted = cart_model.predict(test_fea_vector) 
	
	predicted_id = [[i,p] for i,p in enumerate(predicted)] 
	predicted_sorted = sorted(predicted_id, key=lambda x: x[-1]) 
	# assigning predicted ranks
	predicted_rank_sorted = [[p[0], p[-1], i] for i,p in enumerate(predicted_sorted)]
	rank_diffs = [abs(p[0] - p[-1]) for p in predicted_rank_sorted] 
	return np.mean(rank_diffs) 

def split_data_by_fraction(csv_file, fraction, seed):
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

	return [train_pool, test_pool, validation_pool]

def predict_by_rank_based(train_pool, test_pool):

	# initilize train set
	# rd.shuffle(train_pool) # diff
	train_set = train_pool[:10]
	count = 10
	lives = 3
	last_rd = sys.maxsize

	while lives>=0 and count<len(train_pool):
		train_set.append(train_pool[count])
		count = count + 1

		current_rd = predict_by_cart(train_set, test_pool)

		if current_rd >= last_rd:
			lives = lives - 1
		else:
			lives = 3

		last_rd = current_rd

	return train_set


if __name__ == "__main__":

	#######################################################################################

	# data split
	split_data = split_data_by_fraction("data/Apache_AllMeasurements.csv", 0.4, 0)
	train_pool = split_data[0]
	test_pool = split_data[1]
	validation_pool = split_data[2]

	# apply rank-based method on proj 
	print("### Testing on Test Pool: ")
	train_set = predict_by_rank_based(train_pool, test_pool)
	for config in train_set:
		print(config.index, ",", end="")

	print("\n--------------------")

	# evaluate on validation pool
	rd = predict_by_cart(train_set, validation_pool)
	print("### Evaulation on Validation Pool: ")
	print("[rank difference]:", rd)

	#######################################################################################

	lowest_rank = find_lowest_rank(train_set, validation_pool)
	print("[ming rank]:", lowest_rank)

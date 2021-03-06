#!\usr\bin\python
# coding=utf-8
# Author: youngfeng
# Update: 07/08/2018

"""
Progressive, concluded by Sarkar et al. (ase '15), is one of the basic sampling techiques in performance prediction.
It iteratively randomly select samples from train pool to train a cart model, and test on testing pool untill the 
learning curve come to flatten/convergence point.
The details of Progressive are introduced in paper "Cost-Efficient Sampling for Performance Prediction of Configurable Systems".
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
	"""
	def __init__(self, index, features, perfs, predicted):
		self.index = index
		self.features = features
		self.perfs = perfs
		self.predicted = predicted

def find_lowest_rank(train_set, test_set):
	"""
	build cart model on train_set and predict on test_set, to find the lowest rank in predicted top-10 test_set
	"""
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
	# for sf in select_few[:10]:
	# 	print("actual rank:", sf[0], " actual value:", sorted_test[sf[0]].perfs[-1], " predicted value:", sf[1], " predicted rank: ", sf[2])
	# print("------------")

	return np.min([sf[0] for sf in select_few])


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

	# print("[min samples split]: ", minsplit)
	# print("[min samples leaf] : ", minbucket)

	# cart_model = DecisionTreeRegressor( min_samples_split = minsplit,
	# 									min_samples_leaf = minbucket,
	# 									max_depth = 30)

	cart_model = DecisionTreeRegressor()
	cart_model.fit(train_fea_vector, train_pef_vector)
	test_pef_predicted = cart_model.predict(test_fea_vector)

	mmre_lst = []

	for (config, predicted_perf) in zip(test_set, test_pef_predicted):
		config.predicted[-1] = predicted_perf

	for config in test_set:
		mmre = abs(config.perfs[-1] - config.predicted[-1])/abs(config.perfs[-1])
		mmre_lst.append(mmre)

	return np.mean(mmre_lst)

def split_data_by_fraction(csv_file, fraction):
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
			))

	# for config in configs:
	# 	print(config.index, "-", config.perfs, "-", config.predicted, "-", config.rank)

	# step4: data split
	# fraction = 0.4 # split fraction 
	# rd.seed(seed) # random seed
	rd.shuffle(configs) # shuffle the configs
	indexes = range(len(configs))
	train_index = indexes[:int(fraction*len(configs))]
	test_index = indexes[int((fraction)*len(configs)): int((fraction+0.2)*len(configs))]
	validation_index = indexes[int((fraction+0.2)*len(configs)):]

	train_pool = [configs[i] for i in train_index]
	test_pool = [configs[i] for i in test_index]
	validation_pool = [configs[i] for i in validation_index]

	return [train_pool, test_pool, validation_pool]


def predict_by_progressive(train_pool, test_pool):

	# rd.shuffle(train_pool)
	train_set = train_pool[:10]
	count = 10
	lives = 3
	last_mmre = -1

	# step6: progressive cycle
	while lives > 0 and count < len(train_pool):
		# add sample to train set
		train_set.append(train_pool[count])
		count = count + 1

		current_mmre = predict_by_cart(train_set, test_pool)
		COLLECTOR.append([count, (1-current_mmre)])

		if (1-current_mmre) <= last_mmre:
			lives = lives - 1
		else:
			lives = 3

		last_mmre = (1-current_mmre)

		# if current_mmre < 0.1:
		# 	break

	return train_set


if __name__ == "__main__":

	#######################################################################################

	split_data = split_data_by_fraction("data/Apache_AllMeasurements.csv", 0.4)
	train_pool = split_data[0]
	test_pool = split_data[1]
	validation_pool = split_data[2]

	# apply progressive on proj
	print("### Testing on Test Pool: ")
	train_set = predict_by_progressive(train_pool, test_pool)
	for config in train_set:
		print(config.index, ",", end="")
	
	print("\n--------------------")

	# evaluate on validation pool
	mmre = predict_by_cart(train_set, validation_pool)
	print("### Evaulation on Validation Pool:")
	print("[mmre]:", mmre)

	#######################################################################################
	
	# sort the validation pool by predicted_perf
	rk = find_lowest_rank(train_set, validation_pool)
	print("[min rank]:", rk)


#!\usr\bin\python
# coding=utf-8
# Author: youngfeng
# Update: 07/16/2018

"""
Flash, proposed by Nair et al. (arXiv '18), which aims to find the (near) optimal configuration in unevaluated set.
STEP 1: select 80%% of original data as dataset
STEP 2: split the dataset into training set (30 configs) and unevaluated set (remaining configs)
STEP 3: predict the optimal configuration in unevaluated set, then remove it from unevaluated set to training set.
STEP 4: repeat the STEP 4 until the budget (50 configs) is loss out.
The details of Progressive are introduced in paper "Finding Faster Configurations using FLASH".
"""

import pandas as pd
import random as rd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

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


def remove_by_index(config_pool, index):
	"""
	remove the selected configuration
	"""
	for config in config_pool:
		if config.index == index:
			config_pool.remove(config)
			break

	return config_pool


def find_lowest_rank(train_set, test_set):
	"""
	return the lowest rank in top 10
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
	# for sf in select_few:
	# 	print("actual rank:", sf[0], " actual value:", sorted_test[sf[0]].perfs[-1], " predicted value:", sf[1], " predicted rank:", sf[2])
	# print("-------------")

	return np.min([sf[0] for sf in select_few])


def predict_by_cart(train_set, test_set):
	"""
	return the predicted optimal condiguration
	"""
	train_features = [config.features for config in train_set]
	train_perfs = [config.perfs[-1] for config in train_set]

	test_features = [config.features for config in test_set]

	cart_model = DecisionTreeRegressor()
	cart_model.fit(train_features, train_perfs)
	predicted = cart_model.predict(test_features)

	predicted_id = [[i,p] for i,p in enumerate(predicted)] 
	predicted_sorted = sorted(predicted_id, key=lambda x: x[-1]) # sort test_set by predicted performance

	return test_set[predicted_sorted[0][0]] # the optimal configuration



def split_data_by_fraction(csv_file, fraction):
	"""
	split data set and return the 80% data
	"""
	# step1: read from csv file
	pdcontent = pd.read_csv(csv_file) 
	attr_list = pdcontent.columns # all feature list

	# step2: split attribute - method 1
	features = [i for i in attr_list if "$<" not in i]
	perfs = [i for i in attr_list if "$<" in i]
	sortedcontent = pdcontent.sort_values(perfs[-1]) # from small to big
	# print(len(sortedcontent))
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
	dataset = [configs[i] for i in train_index]
	# print(len(dataset))
	return dataset


def predict_by_flash(dataset, size=30, budget=50):
	"""
	use the budget in dataset to train a best model,
	return the train_set and unevaluated_set
	"""
	#initilize the train set with 30 configurations
	rd.shuffle(dataset)
	train_set = dataset[:size]
	unevaluated_set = dataset

	for config in train_set:
		unevaluated_set = remove_by_index(unevaluated_set, config.index) # remove train_set

	while budget >= 0: # budget equals to 50
		
		optimal_config = predict_by_cart(train_set, unevaluated_set)

		# print("[add]:", optimal_config.index)
		
		unevaluated_set = remove_by_index(unevaluated_set, optimal_config.index)
		
		train_set.append(optimal_config)
		
		budget = budget - 1

	return [train_set, unevaluated_set]

if __name__ == "__main__":

	#######################################################################################

	# select 80% data
	dataset = split_data_by_fraction("data/Apache_AllMeasurements.csv", 0.8)
	print("### initialzation")
	for i in dataset:
		print(str(i.index), ",", end="")
	print("\n-------------")
	data = predict_by_flash(dataset)

	print("### finally split")
	train_set = data[0]
	uneval_set = data[1]
	for i in train_set:
		print(str(i.index), ",", end="")
	print("\n-------------")
	for i in uneval_set:
		print(str(i.index), ",", end="")
	print("\n-------------")

	#######################################################################################

	lowest_rank = find_lowest_rank(train_set, uneval_set)

	print(lowest_rank)


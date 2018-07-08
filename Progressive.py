#!\usr\bin\python
#coding=utf-8
# Author: youngfeng
# Update: 07/08/2018

"""
Progressive, concluded by Sarkar et al. (ase '2015), is one of the basic sampling techiques in performance prediction.
It iteratively randomly select samples from train pool to train a cart model, and test on testing pool untill the 
learning curve come to flatten/convergence point.
The details of CART are introduced in paper "Cost-Efficient Sampling for Performance Prediction of Configurable Systems".
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
	def __init__(self, index, features, perfs):
		self.index = index
		self.features = features
		self.perfs = perfs


def predicted_by_progressive(csv_file, fraction, seed):
	"""
	apply progressive on project in path, we split data into 3 parts, including
	train_pool(fraction), test_pool(0.2), fraction(1-0.2-fraction)
	then return the train_set to build the cart model
	"""
	# step1: read from csv file
	pdcontent = pd.read_csv(csv_file) 
	attr_list = pdcontent.columns # all feature list

	# step2: split attribute - method 1
	features = [i for i in attr_list if "$<" not in i]
	perfs = [i for i in attr_list if "$<" in i]
	pdcontent.sort_values(perfs[-1]) # from small to big

	# step3: collect configuration
	configs = list()
	for c in range(len(pdcontent)):
		configs.append(config_node(c, 
									pdcontent.iloc[c][features].tolist(),
									pdcontent.iloc[c][perfs].tolist(),
			))

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

	DATASETS = [train_pool, test_pool, validation_pool]

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
	last_mmre = -1
	collector = []

	# step6: progressive cycle
	while lives > 0 and count < len(train_pool):
		# add sample to train set
		train_set.append(train_pool[count])
		count = count + 1

		train_fea_vector = [new_sample.features for new_sample in train_set]
		train_pef_vector = [new_sample.perfs[-1] for new_sample in train_set]
		
		# train model based on train set 
		# setting of minsplit and minbucket
		S = len(train_set)
		if S <= 100:
			minbucket = np.floor((S/10)+(1/2))
			minsplit = 2*minbucket
		else:
			minsplit = np.floor((S/10)+(1/2))
			minbucket = np.floor(minsplit/2)

		if minbucket < 2:
			minbucket = 2
		if minsplit < 4:
			minsplit = 4

		minbucket = int(minbucket) # cart cannot set a float minbucket or minsplit 
		minsplit = int(minsplit)

		print("[min samples split]: ", minsplit)
		print("[min samples leaf] : ", minbucket)

		cart_model = DecisionTreeRegressor( min_samples_split = minsplit,
											min_samples_leaf = minbucket)
		cart_model.fit(train_fea_vector, train_pef_vector)
		
		test_fea_vector = [new_sample.features for new_sample in test_pool]
		test_pef_vector = [new_sample.perfs[-1] for new_sample in test_pool]

		# test on test pool
		test_pef_predicted = cart_model.predict(test_fea_vector)
		mmre_lst = []

		for actual, predicted in zip(test_pef_vector, test_pef_predicted):
			mmre = abs(actual-predicted)/abs(actual)
			mmre_lst.append(mmre)

		current_mmre = np.mean(mmre_lst)
		print("[train]: ",count,", [mmre]: ", 1-current_mmre)
		COLLECTOR.append([count, (1-current_mmre)])

		if (1-current_mmre) <= last_mmre:
			lives = lives - 1
		last_mmre = (1-current_mmre)

		# if (1-current_mmre) > 0.9:
		# 	break

	return train_set


if __name__ == "__main__":

	#apply progressive on proj
	train_set = predicted_by_progressive("data/Apache_AllMeasurements.csv", 0.4, 0)
	for sample in train_set:
		print(sample.index, " ", end="")

	# visualize the learning curve
	len_dot = len(COLLECTOR)
	x = []
	y = []

	for i in range(len_dot):
		x.append(COLLECTOR[i][0])
		y.append(COLLECTOR[i][1])

	plt.plot(x, y)
	plt.show()
	

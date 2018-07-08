#!/usr/bin/python
# coding=utf-8
# Author: youngfeng
# Update: 07/08/2018

"""
CART, proposed by Jianmei Guo et al. (ase '13), is one of the state-of-the-art methods in configuration performance prediction.
It simply construct a regression tree to predict the preformance of each configuration.
The details of CART are introduced in paper "Variability-aware performance prediction: statistical learning approach".
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
	def __init__(self, index, features, perfs):
		self.index = index
		self.features = features
		self.perfs = perfs


def predict_by_CART(csv_file, fraction, seed):
	"""
	apply cart on project in csv_file, we split data in 2 parts, including
	train_set(fraction), test_set(1-fraction)
	then return the mmre of this run
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
	# fraction = 0.3 # split fraction 
	rd.seed(seed) # random seed
	rd.shuffle(configs) # shuffle the configs

	indexes = range(len(configs))
	train_index = indexes[:int(fraction*len(indexes))]
	test_index = indexes[int(fraction*len(indexes)):]

	train_set = [configs[i] for i in train_index]
	test_set = [configs[i] for i in test_index]

	# print train_set and test_set
	# for config in train_set:
	# 	print(config.index, " ", end="")
	# print("-------------")
	# for config in test_set:
	# 	print(config.index, " ", end="")

	# step5: model building
	cart_model = DecisionTreeRegressor()
	train_fea_vector = [i.features for i in train_set]
	train_pef_vector = [i.perfs[-1] for i in train_set]
	test_fea_vector = [i.features for i in test_set]
	test_pef_vector = [i.perfs[-1] for i in test_set]

	cart_model.fit(train_fea_vector, train_pef_vector)
	test_pef_predicted = cart_model.predict(test_fea_vector)

	# step6: calculate the mmre
	mmre_lst = []
	for actual, predicted in zip(test_pef_vector, test_pef_predicted):
		mmre = abs(actual-predicted)/abs(actual)
		mmre_lst.append(mmre)
		# print(mmre)

	print("[MMRE]: ", np.mean(mmre_lst))

if __name__ == "__main__":

	projs = ["data/X264_AllMeasurements.csv", 
			 "data/BDBC_AllMeasurements.csv", 
			 "data/SQL_AllMeasurements.csv", 
			 "data/WGet.csv"]

	for proj in projs:
		print("dealing with %s"%proj)
		predict_by_CART(proj, 0.3, 0)
		print("")
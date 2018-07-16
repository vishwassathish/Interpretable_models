import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from data_folds import PrepareData
from sys import argv


class ExplLogisticRegression():

	def __init__(self, penalty="l2", dual=False,tol=0.0001, \
				C=1.0,fit_intercept=True, intercept_scaling=1.0, \
				class_weight=None, random_state=None, \
				solver="liblinear", max_iter=100, \
				multi_class='ovr'):
		
		self.model = linear_model.LogisticRegression(penalty=penalty,\
						dual=dual,tol=tol,C=C, fit_intercept=fit_intercept,\
						intercept_scaling=intercept_scaling, class_weight=class_weight,\
						random_state=random_state, solver=solver,max_iter=max_iter,\
						multi_class=multi_class)

	def train_model(self, x_train, y_train, x_validate=[], y_validate=[]):

		self.model.fit(x_train, y_train)
		print("Done training ...")
		print("Coefficients : ", logObj.coef_, \
			"\nIntercept : ", logObj.intercept_)
		
		if(x_validate != [] and y_validate != []):
			print("Accuracy : ",accuracy_score(y_validate,\
				self.model.predict(x_validate)))

		return self

	def test_model(self, x_validate, y_validate):
		print("Accuracy : ",accuracy_score(y_validate,\
				self.model.predict(x_validate)))
		return self

	def explain_model(self, explaination_type=None):
		## TODO ##
		raise NotImplementedError
 



class ExplRandomForestClassifier():

	def __init__(self, n_estimators=10, criterion="gini",\
				max_features="auto",max_depth=None, \
				min_samples_split=2,min_samples_leaf=1):

		self.model = linear_model.RandomForestClassifier(n_estimators=n_estimators,\
						criterion=criterion,max_features=max_features,\
						max_depth=max_depth,min_samples_split=min_samples_split)
		

	def train_model(self, x_train, y_train, x_validate=[], y_validate=[]):
		raise NotImplementedError

	def test_model(self, x_validate, y_validate):
		raise NotImplementedError

	def explain_model(self, explaination_type=None):
		raise NotImplementedError

	## TODO ##

if __name__ == "__main__" :

	path = argv[1]
	dataObj = PrepareData()
	data = dataObj.read_data(path)
	x_train, y_train, x_validate, y_validate = dataObj.split_train_test(data, split_ratio=0.6)

	## TODO ##

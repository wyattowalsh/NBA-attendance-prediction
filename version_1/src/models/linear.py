import numpy as np
import pandas as pd
from pathlib import Path
import src.data.datasets as ds
import src.data.train_test_split as split
import src.models.metrics as metrics
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV, HuberRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR

def baseline_creation(save = False):
	'''

	'''

	file_names = ['dataset_1', 'dataset_2', 'dataset_3']
	names = ['Dataset 1', "Dataset 2", "Dataset 3"]
	sets = split.split_subsets(file_names)

	df_avgs = pd.DataFrame()
	df_ols = pd.DataFrame()
	for i, file_name in enumerate(file_names):
		y_bar = np.mean(sets[file_name][2])
		preds = np.ones(len(sets[file_name][3])) * y_bar
		avg_score = metrics.apply_metrics('{} Average'.format(names[i]), sets[file_name][3], preds, sets[file_name][2])
		df_avgs = pd.concat([df_avgs, avg_score], axis = 0)
		ols_score = linear(file_name)[1]
		df_ols = pd.concat([df_ols, ols_score], axis = 0)
	
	if save == True:
		to_save_avgs = Path().resolve().joinpath('models', 'baseline', '{}.csv'.format('averages'))
		df_avgs.to_csv(to_save_avgs)
		to_save_ols = Path().resolve().joinpath('models', 'baseline', '{}.csv'.format('OLS'))
		df_ols.to_csv(to_save_ols)

	return df_avgs, df_ols

def collect_all_statistics(save = False):
	"""

	"""
	names = list(ds.get_names().keys())
	df = pd.DataFrame()
	for name in names:
		stats = collect_statistics(name)
		df = pd.concat([df, stats], axis = 0)

	if save:
		to_save = Path().resolve().joinpath('models', 'linear', '{}.csv'.format('performance_outcomes_all'))
		df.to_csv(to_save)	
	return df


def collect_statistics(name):
	'''Runs multiple variations of all linear models and outputs a dataframe with statistics
	

	Variations include: numerical predictor standardization, time series cross-validator, feature selection
	'''

	results = pd.DataFrame()
	models = [linear, ridge, lasso, elastic_net, huber, support_vector_machine]
	for model in models:
		results = pd.concat([results, model(name)[1]], axis = 0)
	return results

def linear(name):
	'''Outputs a fitted Linear Regression Model.

	Inputs can be standardized or not
	'''

	display_name = ds.get_names()[name]
	X_train, X_test, y_train, y_test, train = split.split_subset(name, True)
	model = LinearRegression().fit(X_train, y_train)
	performance = metrics.apply_metrics('{} OLS'.format(display_name), y_test, model.predict(X_test),y_train)
	performance['Tuning Parameters'] = ""
	return model, performance

def ridge(name, cv = 5):
	'''Outputs a fitted Ridge Regression Model with a penalty term tuned through cross validation.

	'''
	
	display_name = ds.get_names()[name]
	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	X_train, X_test= split.standardize(X_train, X_test)
	alphas = np.linspace(500, 750, 50)
	model = RidgeCV(alphas=alphas, fit_intercept=True, cv=cv).fit(X_train, y_train)
	performance = metrics.apply_metrics('{} Ridge'.format(display_name), y_test, model.predict(X_test),y_train)
	performance['Tuning Parameters'] = [{'Alpha': model.alpha_}]
	return model, performance

def lasso(name, cv = 5):
	'''Outputs a fitted Lasso Regression Model with a penalty term tuned through cross validation.

	Inputs must be standardized.
	Number of folds in cross validation is by default 5.
	n_jobs = -1 allows for all local processors to be utilized.
	'''
	display_name = ds.get_names()[name]
	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	X_train, X_test= split.standardize(X_train, X_test)
	model = LassoCV(n_alphas=50, verbose = 0, cv=5, n_jobs=-1, copy_X = True, tol = 1e-3,
	random_state = 18).fit(X_train, y_train)
	performance = metrics.apply_metrics('{} Lasso'.format(display_name), y_test, model.predict(X_test),y_train)
	performance['Tuning Parameters'] = [{'Alpha': model.alpha_}]
	params = model.coef_
	return model, performance, params

def elastic_net(name, cv = 5):
	'''Outputs a fitted Elastic Net Regression Model with tuning parameters found through cross validation.
	
	Inputs must be standardized.
	l1_ratios are spread out on a log scale as recommended by package authors.
	Number of folds in cross validation is by default 5.
	n_jobs = -1 allows for all local processors to be utilized.
	# '''
	# if np.any(X_train.mean(axis = 0) > 1):
	# 	raise ValueError('Numerical features must be standardized')

	display_name = ds.get_names()[name]
	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	X_train, X_test= split.standardize(X_train, X_test)
	l1_ratios = np.geomspace(1e-8,1,50)
	model = ElasticNetCV(l1_ratio=l1_ratios, n_alphas=50, cv = 5, verbose = 0, 
	                     n_jobs = -1, random_state = 18).fit(X_train, y_train)

	performance = metrics.apply_metrics('{} Elastic Net'.format(display_name), y_test, model.predict(X_test),y_train)
	performance['Tuning Parameters'] = [{'Alpha': model.alpha_,'L1 Ratio': model.l1_ratio_}]
	return model, performance

def huber(name, cv = 5):
	'''

	'''

	display_name = ds.get_names()[name]
	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	X_train, X_test= split.standardize(X_train, X_test)
	to_score, scoring = metrics.create_metrics()
	param_grid = {'epsilon': np.linspace(1+1e-15, 1.2, 10),
	'alpha': np.linspace(1e-8, 2, 10)}
	model = HuberRegressor()
	model_cv = GridSearchCV(model, param_grid= param_grid, scoring = 'neg_mean_absolute_error', 
	                        n_jobs = -1, pre_dispatch = 16, cv = cv,
	                        refit = True).fit(X_train, y_train)
	performance = metrics.apply_metrics('{} Huber'.format(display_name), y_test, model_cv.predict(X_test),y_train)
	performance['Tuning Parameters'] = [model_cv.best_params_]

	return model_cv, performance

def support_vector_machine(name, cv = 5):
	"""

	"""

	display_name = ds.get_names()[name]
	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	X_train, X_test = split.standardize(X_train, X_test)
	to_score, scoring = metrics.create_metrics()
	param_grid = { 
	'epsilon': np.linspace(-2,2,4),
	'fit_intercept': [True],
	'C': np.linspace(1e6,1e10, 50),
	'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
	'dual': [False], 'random_state': [18]}
	model = LinearSVR()
	model_cv = GridSearchCV(model, param_grid= param_grid, scoring = 'neg_mean_absolute_error', 
	                        n_jobs = -1,  pre_dispatch = 16, cv = cv, 
	                        refit = True).fit(X_train, y_train)
	performance = metrics.apply_metrics('{} Linear Support Vector Machine'.format(display_name), 
	                                    y_test, model_cv.predict(X_test),y_train)
	performance['Tuning Parameters'] = [model_cv.best_params_]
	return model_cv, performance







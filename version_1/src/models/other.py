import numpy as np
import pandas as pd
import src.data.train_test_split as split
import src.data.datasets as ds
import src.models.linear as linear
import src.models.metrics as metrics
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

def get_all_results(save=False):
	'''

	'''
	model_names = ['k_neighbors', 'svr']
	results = pd.DataFrame()
	for name in model_names:
		df = pd.read_csv(Path().resolve().joinpath('models', 'other', '{}.csv'.format(name)), index_col = 0)
		results = pd.concat([results, df], axis = 0)

	if save:
		to_save = Path().resolve().joinpath('models', 'other', 'performance_outcomes_all.csv')
		results.to_csv(to_save)

	return results

def get_results(model_name, save = False):
	'''

	'''
	names = ['dataset_3']
	models = [k_neighbors_grid_cv, svr_grid_cv]
	model_names = ['k_neighbors', 'svr']
	model_dict = dict(zip(model_names, models)) 
	model = model_dict[model_name]
	results = pd.DataFrame()
	for name in names:
		result_k_fold = model(name = name)[1]
		result_tss = model(name, cv = TimeSeriesSplit(5))[1]
		results = pd.concat([results,result_k_fold, result_tss], axis = 0)

	if save:
		to_save = Path().resolve().joinpath('models', 'other', '{}.csv'.format(model_dict[model]))
		results.to_csv(to_save)

	return results

def get_k_neighbors_randomized_results(tss = False):
	'''

	'''

	names = ['dataset_1', 'dataset_2', 'dataset_3'] 
	results_dict = {}
	results_df = {}
	if tss:
		for name in names:
			results_dict[name] = k_neighbors_randomized_cv(name, cv = TimeSeriesSplit(5)).cv_results_
			results_df[name] = pd.DataFrame.from_dict(results_dict[name])
			to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
												'other', 'k_neighbors',
												'{}_time_series_split.csv'.format(name))
			results_df[name].to_csv(to_save)

		return results_df
	else:
		for name in names:
			results_dict[name] = k_neighbors_randomized_cv(name).cv_results_
			results_df[name] = pd.DataFrame.from_dict(results_dict[name])
			to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
												'other', 'k_neighbors', 
												'{}_k_fold.csv'.format(name))
			results_df[name].to_csv(to_save)

		return results_df

def k_neighbors_randomized_cv(name, n_iter = 50, cv = 5):
	"""

	"""

	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	X_train = split.standardize(name, X_train)
	X_test = split.standardize(name, X_test)
	to_score = metrics.create_metrics()[0]
	param_grid = {'n_neighbors': np.arange(2,50,2, dtype = int),
	'weights': ['uniform', 'distance'],
	'leaf_size': [2,4,8,16,32,64,128, 256]}  

	model = KNeighborsRegressor(n_jobs = -1)
	model_cv = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs = -1, pre_dispatch = 16,
								  n_iter= n_iter, cv=cv, scoring= to_score,
								  refit = False, random_state = 18).fit(X_train, y_train)

	return model_cv


def k_neighbors_grid_cv(name, cv = 5, save = True):
	'''Conducts a grid search over all possible combinations of given parameters and returns result.

	Uses parameters closely clustered around the best randomized search results.
	Also returns back best fitted model by specified criteria (MAE).
	'''

	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	X_train, X_test = split.standardize(name, X_train, X_test)
	to_score = metrics.create_metrics()[0]
	param_grid = {'n_neighbors': np.arange(20,51,2, dtype = int),
	'weights': ['distance'],
	'leaf_size': [8,16, 128, 256]} 
				   
	model = KNeighborsRegressor(n_jobs = -1)
	model_cv = GridSearchCV(n_jobs = -1, estimator=model, param_grid=param_grid, scoring = to_score, pre_dispatch = 16,
						 refit = False, cv = cv).fit(X_train, y_train)

	display_name = ds.get_names()[name]
	performance = pd.DataFrame()
	variations = linear.get_model_variants(KNeighborsRegressor, model_cv)
	for variation in variations:
		model = variation.fit(X_train, y_train).predict(X_test)
		performance = pd.concat([performance, metrics.apply_metrics('{} K Neighbors'.format(display_name),
	 y_test, model)], axis = 0)

	if save:
		to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes',
		'other', 'k_neighbors','{}.csv'.format('grid'))
		results = pd.DataFrame.from_dict(model_cv.cv_results_)
		results.to_csv(to_save)


	return model_cv, performance

def get_svr_randomized_results(tss = False):
	'''

	'''

	names = ['dataset_1', 'dataset_2', 'dataset_3'] 
	results_dict = {}
	results_df = {}
	if tss:
		for name in names:
			results_dict[name] = svr_randomized_cv(name, cv = TimeSeriesSplit(5)).cv_results_
			results_df[name] = pd.DataFrame.from_dict(results_dict[name])
			to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
													'other', 'svr',
													'{}_time_series_split.csv'.format(name))
			results_df[name].to_csv(to_save)

		return results_df
	else:
		for name in names:
			results_dict[name] = svr_randomized_cv(name).cv_results_
			results_df[name] = pd.DataFrame.from_dict(results_dict[name])
			to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
												'other', 'svr', 
												'{}_k_fold.csv'.format(name))
			results_df[name].to_csv(to_save)

		return results_df

def svr_randomized_cv(name, n_iter = 25, cv = 5):
	"""

	"""

	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	X_train = split.standardize(name, X_train)
	X_test = split.standardize(name, X_test)
	to_score = metrics.create_metrics()[0]
	param_grid = {'kernel': ['poly', 'rbf'],
	'degree': np.arange(2,6),
	'gamma': ['scale', 'auto'] ,
	'C': np.linspace(1e-5, 5, 20),
	'epsilon': np.linspace(0,1,20),
	'shrinking' : [True, False]} 

	model = SVR()
	model_cv = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs = -1, pre_dispatch = 16,
								  n_iter= n_iter, cv=cv, scoring= to_score, 
								  random_state = 18, refit = False).fit(X_train, y_train)

	return model_cv

def svr_grid_cv(name, standardize = False, cv = 5):
	"""

	"""

	if cv == 5:
		cv_type = 'K-Fold'
	else:
		cv_type = "Time Series Split"

	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	if standardize:
		X_train = split.standardize(name, X_train)
		X_test = split.standardize(name, X_test)
	to_score = metrics.create_metrics()[0]
	param_grid = {'kernel': ['poly', 'rbf', 'sigmoid'],
	'degree': np.arange(3,9),
	'gamma': ['scale', 'auto'] ,
	'C': [2e-5,2e-3,2e-1,2e1,2e3,2e5,2e7,2e9,2e11]} 

	model = SVR(n_jobs = -1)
	model_cv = GridSearchCV(estimator=model, param_distributions=param_grid, n_jobs = -1, pre_dispatch = 16,
							 refit= False, cv=cv, scoring= to_score).fit(X_train, y_train)

	display_name = ds.get_names()[name]
	performance = pd.DataFrame()
	variations = linear.get_model_variants(KNeighborsRegressor, model_cv)
	for variation in variations:
		model = variation.fit(X_train, y_train).predict(X_test)
		performance = pd.concat([performance, metrics.apply_metrics('{} {} Support Vector Machine'.format(display_name, cv_type),
	 y_test, model)], axis = 0)

	return model_cv, performance


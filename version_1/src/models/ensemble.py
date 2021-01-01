import numpy as np
import pandas as pd
import src.data.train_test_split as split
import src.data.datasets as ds
import src.models.linear as linear
import src.models.metrics as metrics
from pathlib import Path
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

def get_all_results(save=False):
	'''

	'''
	model_names = ['random_forest', 'adaboost', 'gradient_boosting', 'extra_trees']
	results = pd.DataFrame()
	for name in model_names:
		df = pd.read_csv(Path().resolve().joinpath('models', 'ensemble', '{}.csv'.format(name)), index_col = 0)
		results = pd.concat([results, df], axis = 0)

	if save:
		to_save = Path().resolve().joinpath('models', 'ensemble', 'performance_outcomes_all.csv')
		results.to_csv(to_save)

	return results

def get_results(model_name, save = False):
	'''

	'''

	names = ['dataset_1', 'dataset_2','dataset_3']
	models = [random_forest_grid_cv, adaboost_grid_cv, gradient_boosting_randomized_cv, extra_trees_grid_cv]
	model_names = ['random_forest', 'adaboost', 'gradient_boosting', 'extra_trees']
	model_dict = dict(zip(model_names, models))	
	model = model_dict[model_name]
	results = pd.DataFrame()
	for name in names:
		result_k_fold = model(name = name)[1] 
		results = pd.concat([results,result_k_fold], axis = 0)

	if save:
		to_save = Path().resolve().joinpath('models', 'ensemble', '{}.csv'.format(model_name))
		results.to_csv(to_save)

	return results

def get_baselines(save = False):
	'''

	''' 

	names = ['dataset_1','dataset_2', 'dataset_3']
	models = [RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor]
	model_names = ['random_forest', 'adaboost', 'gradient_boosting', 'extra_trees']
	model_dict = dict(zip(models, model_names))	
	display_name = ds.get_names()
	results = pd.DataFrame()
	for name in names:
		X_train, X_test, y_train, y_test, train = split.split_subset(name)
		disp_name = display_name[name]
		for func in models:
			preds = func().fit(X_train,y_train).predict(X_test)
			performance = metrics.apply_metrics('{} {}'.format(disp_name, model_dict[func]), y_test, preds)
			results = pd.concat([results, performance], axis = 0)

	if save:
		to_save = Path().resolve().joinpath('models', 'ensemble', '{}.csv'.format('baseline_all'))
		results.to_csv(to_save)

	return results


def get_rf_randomized_results(tss = False):
	'''

	'''

	names = ['dataset_1', 'dataset_2', 'dataset_3'] 
	results_dict = {}
	results_df = {}
	if tss:
		for name in names:
		    results_dict[name] = random_forest_randomized_cv(name, cv = TimeSeriesSplit(5)).cv_results_
		    results_df[name] = pd.DataFrame.from_dict(results_dict[name])
		    to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
		                                        'ensemble', 'random_forest', 
		                                        '{}_time_series_split.csv'.format(name))
		    results_df[name].to_csv(to_save)

		return results_df
	else:
		for name in names:
		    results_dict[name] = random_forest_randomized_cv(name).cv_results_
		    results_df[name] = pd.DataFrame.from_dict(results_dict[name])
		    to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
		                                        'ensemble', 'random_forest', 
		                                        '{}_k_fold.csv'.format(name))
		    results_df[name].to_csv(to_save)

		return results_df

def random_forest_randomized_cv(name, n_iter = 30, cv = 5):
	'''Conducts a randomized search of cross validation for given parameters of the random forest and returns results.

	Implements scoring criteria based off of custom dictionary.
	'''

	X_train, X_test, y_train, y_test, train = split.split_subset(name)

	random_grid_0 = {'n_estimators': np.linspace(start=100, stop= 500, num=20, dtype=int),
	'max_depth' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
	'bootstrap': [True, False],
	'max_features': np.linspace(2, len(X_train.columns), num = 20, dtype = int),
	'min_samples_split': [2, 4, 8, 16, 32, 64, 128],
	'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64]}

	rf = RandomForestRegressor()
	rf_cv = RandomizedSearchCV(estimator=rf, param_distributions= random_grid_0, n_jobs = -1, n_iter= n_iter, cv=cv,
	pre_dispatch = 16, scoring= 'neg_mean_absolute_error', random_state = 18, refit = True).fit(X_train, y_train)

	return rf_cv

def random_forest_grid_cv(name, cv = 5, save = True):
	'''Conducts a grid search over all possible combinations of given parameters and returns result.

	Uses parameters closely clustered around the best randomized search results.
	Also returns back best fitted model by specified criteria (MAE).
	'''

	display_name = ds.get_names()[name]
	X_train, X_test, y_train, y_test, train = split.split_subset(name)

	param_grid = {'n_estimators': [400],
	'max_depth' : [30, 50, 90],
	'bootstrap': [False],
	'max_features': [30, 40, 50],
	'min_samples_split': [4],
	'min_samples_leaf': [1, 2, 8]}
				   
	rf = RandomForestRegressor()
	rf_cv = GridSearchCV(n_jobs = -1, estimator=rf, param_grid=param_grid, scoring = 'neg_mean_absolute_error',
	pre_dispatch = 16, refit = True, cv = cv).fit(X_train, y_train)

	performance = metrics.apply_metrics('{} Random Forest'.format(display_name),
	 y_test, rf_cv.predict(X_test), y_train)
	performance['Tuning Parameters'] = [rf_cv.best_params_]

	if save:
		to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes',
		'ensemble', 'random_forest', '{}.csv'.format('grid'))
		results = pd.DataFrame.from_dict(rf_cv.cv_results_)
		results.to_csv(to_save)

	return rf_cv, performance 

def get_ada_randomized_results(tss =  False):
	'''

	'''

	names = ['dataset_1', 'dataset_2', 'dataset_3'] 
	results_dict = {}
	results_df = {}
	if tss:
		for name in names:
		    results_dict[name] = adaboost_randomized_cv(name, cv = TimeSeriesSplit(5)).cv_results_
		    results_df[name] = pd.DataFrame.from_dict(results_dict[name])
		    to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
		                                        'ensemble', 'adaboost',
		                                        '{}_time_series_split.csv'.format(name))
		    results_df[name].to_csv(to_save)

		return results_df
	else:
		for name in names:
		    results_dict[name] = adaboost_randomized_cv(name).cv_results_
		    results_df[name] = pd.DataFrame.from_dict(results_dict[name])
		    to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
		                                        'ensemble', 'adaboost', 
		                                        '{}_k_fold.csv'.format(name))
		    results_df[name].to_csv(to_save)

		return results_df

def adaboost_randomized_cv(name, n_iter = 30, cv = 5):
	"""Conducts a randomized search of cross validation for given parameters of AdaBoost and returns results.

	"""

	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	to_score = metrics.create_metrics()[0]

	param_grid = {'base_estimator': [DecisionTreeRegressor(max_depth=2),
	DecisionTreeRegressor(max_depth=3), 
	DecisionTreeRegressor(max_depth=4),
	DecisionTreeRegressor(max_depth=5)],
	'n_estimators': np.linspace(50, 500, 20, dtype=int),
	'loss': ['linear', 'square', 'exponential'],
	'learning_rate':  np.append(np.array([0]), np.geomspace(1e-3,5, 10))} 

	adaboost = AdaBoostRegressor()
	adaboost_cv = RandomizedSearchCV(estimator=adaboost, param_distributions = param_grid, 
	                                 n_iter = n_iter, n_jobs=-1, pre_dispatch = 16, cv=cv, 
	                                 refit=False, random_state = 18,
									 scoring = 'neg_mean_absolute_error').fit(X_train, y_train)


	return adaboost_cv

def adaboost_grid_cv(name, cv = 5, save = True):
	'''Conducts a grid search over all possible combinations of given parameters and returns the result.

	Uses parameters closely clustered around the best randomized search results.
	'''

	X_train, X_test, y_train, y_test, train = split.split_subset(name)

	param_grid = {'base_estimator': [DecisionTreeRegressor(max_depth=5)],
	'n_estimators': [250],
	'loss': ['linear', 'exponential'],
	'learning_rate':  np.geomspace(1e-6,0.2, 20)} 

	adaboost = AdaBoostRegressor()
	adaboost_cv = GridSearchCV(estimator=adaboost, param_grid=param_grid, scoring = 'neg_mean_absolute_error', 
						 refit = True, cv = cv, n_jobs=-1, pre_dispatch = 16).fit(X_train, y_train)

	display_name = ds.get_names()[name]
	performance = metrics.apply_metrics('{} AdaBoost'.format(display_name), y_test,
	adaboost_cv.predict(X_test), y_train)
	performance['Tuning Parameters'] = [adaboost_cv.best_params_]

	if save:
		to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes',
		'ensemble', 'adaboost','{}.csv'.format('grid'))
		results = pd.DataFrame.from_dict(adaboost_cv.cv_results_)
		results.to_csv(to_save)

	return adaboost_cv, performance 

def get_gb_randomized_results(tss = False):
	'''

	'''

	names = ['dataset_1', 'dataset_2', 'dataset_3'] 
	results_dict = {}
	results_df = {}
	if tss:
		for name in names:
		    results_dict[name] = gradient_boosting_randomized_cv(name, cv= TimeSeriesSplit(5)).cv_results_
		    results_df[name] = pd.DataFrame.from_dict(results_dict[name])
		    to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
		                                        'ensemble', 'gradient_boosting',
		                                        '{}_time_series_split.csv'.format(name))
		    results_df[name].to_csv(to_save)

		return results_df
	else:
		for name in names:
		    results_dict[name] = gradient_boosting_randomized_cv(name).cv_results_
		    results_df[name] = pd.DataFrame.from_dict(results_dict[name])
		    to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
		                                        'ensemble', 'gradient_boosting',
		                                        '{}_k_fold.csv'.format(name))
		    results_df[name].to_csv(to_save)

		return results_df

def gradient_boosting_randomized_cv(name, n_iter = 50, cv =5, save = True):
	"""Conducts a randomized search of cross validation for given parameters of Gradient Boosting and returns results.

	"""	

	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	param_grid = {'loss' : ['ls', 'lad', 'huber'] ,
	'learning_rate': np.append(np.array([0]), np.geomspace(1e-6,1, 50)),
	'n_estimators': np.linspace(500, 1000, 50, dtype=int),
	'min_samples_split': [2, 4, 8, 16, 32, 64, 128, 256],
	'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64, 128],
	'max_depth': [2,3,4,5,10, 15],
	'alpha': np.linspace(1e-6, 1, 25),
	'max_features': np.linspace(2, len(X_train.columns), num = 50, dtype = int),}

	gradient_boosting = GradientBoostingRegressor()
	gradient_boosting_cv = RandomizedSearchCV(estimator= gradient_boosting, n_jobs = -1, pre_dispatch = 16,
	                                          param_distributions = param_grid, n_iter = n_iter, cv=cv, 
	                                          refit=True,
	                                          scoring = 'neg_mean_absolute_error').fit(X_train, y_train)

	display_name = ds.get_names()[name]
	performance = metrics.apply_metrics('{} Gradient Boosting'.format(display_name),
	 y_test, gradient_boosting_cv.predict(X_test) , y_train)
	performance['Tuning Parameters'] = [gradient_boosting_cv.best_params_]


	if save:
		to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes',
		'ensemble', 'gradient_boosting','{}.csv'.format('randomized'))
		results = pd.DataFrame.from_dict(gradient_boosting_cv.cv_results_)
		results.to_csv(to_save)

	return gradient_boosting_cv, performance

def gradient_boosting_grid_cv(name, cv = 5, save = True):
	"""Conducts a grid search over all possible combinations of given parameters and returns the result

	Uses parameters closely clustered around the best randomized search results.
	Also returns back best fitted model by specified criteria (MAE).
	"""

	X_train, X_test, y_train, y_test, train = split.split_subset(name)

	param_grid = {'loss' : ['ls', 'lad', 'huber'] ,
	'learning_rate': np.geomspace(1e-6, 0.1, 5),
	'n_estimators': [900],
	'min_samples_split': [4, 64, 128, 256],
	'min_samples_leaf': [8, 128],
	'max_depth': [4,5,15],
	'alpha': np.linspace(0.1, 1, 5),
	'max_features': [3,40,60]}


	gradient_boosting = GradientBoostingRegressor(random_state = 18)
	gradient_boosting_cv = GridSearchCV(n_jobs = -1, estimator= gradient_boosting, param_grid = param_grid,
										 cv= cv, refit = True, scoring = 'neg_mean_absolute_error', 
										 pre_dispatch = 16).fit(X_train, y_train)

	display_name = ds.get_names()[name]
	performance = metrics.apply_metrics('{} Gradient Boosting'.format(display_name),
	 y_test, gradient_boosting_cv.predict(X_test) , y_train)
	performance['Tuning Parameters'] = [gradient_boosting_cv.best_params_]

	if save:
		to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes',
		'ensemble', 'gradient_boosting','{}.csv'.format('grid'))
		results = pd.DataFrame.from_dict(gradient_boosting_cv.cv_results_)
		results.to_csv(to_save)


	return gradient_boosting_cv, performance 

def get_et_randomized_results(tss = False):
	'''

	'''

	names = ['dataset_1', 'dataset_2', 'dataset_3'] 
	results_dict = {}
	results_df = {}
	if tss:
		for name in names:
		    results_dict[name] = extra_trees_randomized_cv(name, cv = TimeSeriesSplit(5)).cv_results_
		    results_df[name] = pd.DataFrame.from_dict(results_dict[name])
		    to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
		                                        'ensemble', 'extra_trees',
		                                        '{}_time_series_split.csv'.format(name))
		    results_df[name].to_csv(to_save)

		return results_df
	else:
		for name in names:
		    results_dict[name] = extra_trees_randomized_cv(name).cv_results_
		    results_df[name] = pd.DataFrame.from_dict(results_dict[name])
		    to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes', 
		                                        'ensemble', 'extra_trees',
		                                        '{}_k_fold.csv'.format(name))
		    results_df[name].to_csv(to_save)

		return results_df

def extra_trees_randomized_cv(name, n_iter = 30, cv = 5):
	"""

	"""

	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	to_score = metrics.create_metrics()[0]
	extra_trees = ExtraTreesRegressor(random_state = 18, n_jobs = -1, max_features= None, bootstrap = False)

	random_grid = {'n_estimators': [np.linspace(start=100, stop= 500, num=20, dtype=int)],
	'max_depth' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
	'bootstrap': [True, False],
	'max_features': np.linspace(2, len(X_train.columns), num = 20, dtype = int),
	'min_samples_split': [2, 4, 8, 16, 32, 64, 128],
	'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64]}


	extra_trees_cv = RandomizedSearchCV(estimator=extra_trees, param_distributions=random_grid,
										n_iter=n_iter, cv=cv, n_jobs=-1, pre_dispatch = 16, 
										refit=False, 
										scoring ='neg_mean_absolute_error').fit(X_train, y_train)

	return extra_trees_cv

def extra_trees_grid_cv(name, cv = 5, save = True):
	"""

	"""

	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	extra_trees = ExtraTreesRegressor(n_jobs = -1, random_state = 18, max_features= None, bootstrap = False)

	param_grid = {'n_estimators': [250],
	'max_depth' : [20,35],
	'bootstrap': [True, False],
	'max_features': [30, 45, 80],
	'min_samples_split': [2,8,16],
	'min_samples_leaf': [1, 2]}

	extra_trees_cv = GridSearchCV(n_jobs = -1, estimator= extra_trees, param_grid = param_grid, pre_dispatch = 16,
	 cv= cv, refit = True, scoring = 'neg_mean_absolute_error').fit(X_train, y_train)

	display_name = ds.get_names()[name]
	performance = metrics.apply_metrics('{} Extra Trees'.format(display_name),
	 y_test, extra_trees_cv.predict(X_test), y_train)
	performance['Tuning Parameters'] = [extra_trees_cv.best_params_]

	if save:
		to_save = Path().resolve().joinpath('models', 'cross_validation_outcomes',
		'ensemble', 'extra_trees','{}.csv'.format('grid'))
		results = pd.DataFrame.from_dict(extra_trees_cv.cv_results_)
		results.to_csv(to_save)



	return extra_trees_cv, performance







import ast
import pandas as pd
import numpy as np
import src.data.datasets as ds
import src.data.train_test_split as split
import src.models.linear as linear
import src.models.metrics as metrics
from pathlib import Path
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def single_layer_network_randomized_cv(name, n_iter = 20, cv = 5, save = True):
	
	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	X_train, X_test = split.standardize(X_train, X_test)
	param_grid = {'shape' : [(X_train.shape[1],)],
	'batch_size': [256,512,1028,2056],
	'epochs': [25,50],
	'learning_rate':[1e-4,1e-3,1e-2,1e-1,1,10]}
	# 'reg': np.linspace(1e-4, 750, 500),
	# 'if_reg': [True],
	# 'shuffle': [False,True]}
  #kernel_regularizer=regularizers.l2(reg))


	def single_layer_network(shape, learning_rate):
		"""

		"""
	
		net = Sequential()
		net.add(Dense(45, activation='relu', input_shape= shape))
		net.add(Dense(1, activation='linear'))
		optimizer = Adam(learning_rate = learning_rate)
		net.compile(optimizer=optimizer, loss='mean_squared_error')
		return net


	net = KerasRegressor(build_fn = single_layer_network, verbose = 0, workers=8, use_multiprocessing=True )
	net_cv = RandomizedSearchCV(estimator = net, param_distributions = param_grid, n_jobs = -1, pre_dispatch = 16, 
						  refit= True, cv = cv, scoring = 'neg_mean_absolute_error', n_iter = n_iter, 
						  random_state = 18).fit(X_train, y_train)


	display_name = ds.get_names()[name]
	performance = metrics.apply_metrics('{} Single Layer Neural Network'.format(display_name),
	y_test, net_cv.predict(X_test) , y_train)
	performance['Tuning Parameters'] = [net_cv.best_params_]

	
	if save:
		to_save_cv = Path().resolve().joinpath('models', 'cross_validation_outcomes',
		'neural_network', '{}.csv'.format(name))
		results = pd.DataFrame.from_dict(net_cv.cv_results_)
		results.to_csv(to_save_cv)
		to_save_perf = Path().resolve().joinpath('models', 'neural_network', '{}_performance.csv'.format(name))
		performance.to_csv(to_save_perf)

	return net_cv, performance

def single_layer_network_grid_cv(name, cv = 5, save = True):
	
	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	X_train, X_test = split.standardize(X_train, X_test)

	param_grid = {'shape' : [(X_train.shape[1],)],
	'neurons': np.arange(230),
	'batch_size': [1028],
	'epochs': [50],
	'reg': np.geomspace(1e-4, 2, 25),
	'if_reg': [True],
	'shuffle': [True]}

	# param_grid = {'shape' : [(X_train.shape[1],)],
	# 'neurons': np.arange(20,275,5),
	# 'batch_size': [1028],
	# 'epochs': [50],
	# 'learning_rate': np.linspace(0.1,1,50),
	# 'reg': np.geomspace(1e-8, 2, 50),
	# 'if_reg': [True],
	# 'shuffle': [False, True]}



	def single_layer_network(shape, learning_rate, reg, if_reg):
		"""

		"""
	
		net = Sequential()
		if if_reg:
			net.add(Dense(45, activation='relu', input_shape= shape, kernel_regularizer=regularizers.l2(reg)))
		else:
			net.add(Dense(45, activation='relu', input_shape= shape))
		net.add(Dense(1, activation='linear'))
		optimizer = Adam(learning_rate = learning_rate)
		net.compile(optimizer=optimizer, loss='mean_squared_error')
		return net


	net = KerasRegressor(build_fn = single_layer_network, verbose = 0)
	net_cv = GridSearchCV(estimator = net, param_grid = param_grid, n_jobs = -1, pre_dispatch = 16, 
						  refit= True, cv = cv, scoring = 'neg_mean_absolute_error').fit(X_train, y_train)

	display_name = ds.get_names()[name]
	performance = metrics.apply_metrics('{} Single Layer Neural Network'.format(display_name),
	y_test, net_cv.predict(X_test) , y_train)
	performance['Tuning Parameters'] = [net_cv.best_params_]



	return net_cv

def single_layer_network_performance(name, save = True):
	"""

	"""

	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	X_train, X_test = split.standardize(name, X_train, X_test)

	def single_layer_network(neurons, shape, learning_rate, reg, if_reg, shuffle, epochs, batch_size, X_train = X_train, y_train = y_train):
		"""

		"""
	
		net = Sequential()
		if if_reg:
			net.add(Dense(neurons, activation='relu', input_shape= shape, kernel_regularizer=regularizers.l2(reg)))
		else:
			net.add(Dense(neurons, activation='relu', input_shape= shape))
		net.add(Dense(1, activation='linear'))
		optimizer = Adam(learning_rate = learning_rate)
		net.compile(optimizer=optimizer, loss='mean_squared_error')
		net.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, 
		        workers=8, use_multiprocessing=True, shuffle = shuffle, validation_split = 0)
		return net

	results = pd.read_csv(Path().resolve().joinpath('models', 'cross_validation_outcomes', 
	                                                'neural_network', '{}_{}.csv'.format(name, 
	                                                                                     'randomized_3_10_fold')), index_col = 0)

	bestr2 = ast.literal_eval(results.loc[results['rank_test_$R^2$'] == 1, 'params'].values[0])
	bestmae = ast.literal_eval(results.loc[results['rank_test_Mean Absolute Error'] == 1, 'params'].values[0])
	bestrmse = ast.literal_eval(results.loc[results['rank_test_Root Mean Square Error'] == 1, 'params'].values[0])

	display_name = ds.get_names()[name]
	dict_list = [bestr2,bestmae, bestrmse]
	unique_dict_list = [dict(t) for t in {tuple(sorted(d.items())) for d in dict_list}]
	performance = pd.DataFrame()
	for item in unique_dict_list:
		preds = single_layer_network(**item).predict(X_test)
		performance = pd.concat([performance, metrics.apply_metrics('{} Single Layer NN'.format(display_name),
		 y_test, preds)], axis = 0)

	if save:
		to_save = Path().resolve().joinpath('models', 'neural_network',
		'{}.csv'.format('single_layer_network'))
		performance.to_csv(to_save)

	return performance

def multi_layer_network_randomized_cv(name, n_iter = 30, cv = 5, save = True):
	
	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	X_train, X_test = split.standardize(name, X_train, X_test)
	to_score = metrics.create_metrics()[0]
	reg = list(np.geomspace(1e-6, 5, 49))
	param_grid = {'shape' : [(X_train.shape[1],)],
	'neurons_l1': np.arange(5,275,5),
	'neurons_ol': [range(5,275,5), range(5,275,5), range(5,275,5), range(5,275,5), range(5,275,5)],
	'batch_size': [4, 8,16,32,64,128,256,512,1028],
	'epochs': [25,50,100],
	'num_layers': [0,1,2,3,4,5],
	'learning_rate': np.linspace(0.1,1,20),
	'reg_l1': np.append(np.array([0]), np.geomspace(1e-6, 5, 49)),
	'reg_ol': [[0] + reg, [0] + reg, [0] + reg, [0] + reg, [0] + reg, [0] + reg], 
	'shuffle': [False, True]}



	def multi_layer_network(neurons_l1,neurons_ol, num_layers, 
	                        shape, learning_rate, reg_l1, reg_ol):
		"""

		"""
	
		net = Sequential()
		net.add(Dense(neurons_l1, activation='relu', input_shape= shape, kernel_regularizer=regularizers.l2(reg_l1)))
		for i in np.arange(num_layers):
			net.add(Dense(neurons_ol[i], activation='relu', input_shape= shape, 
			              kernel_regularizer=regularizers.l2(reg_ol[i])))
		net.add(Dense(1, activation='linear'))
		optimizer = Adam(learning_rate = learning_rate)
		net.compile(optimizer=optimizer, loss='mean_squared_error')
		return net

	net = KerasRegressor(build_fn = multi_layer_network, verbose = 0)
	net_cv = RandomizedSearchCV(estimator = net, param_distributions = param_grid, n_jobs = -1, pre_dispatch = 16, 
						  refit= False, cv = cv, scoring = to_score, n_iter = n_iter).fit(X_train, y_train)

	if save:
		to_save_cv = Path().resolve().joinpath('models', 'cross_validation_outcomes',
		'neural_network', '{}_{}_{}.csv'.format(name, 'multi_layer_network', 'randomized'))
		results = pd.DataFrame.from_dict(net_cv.cv_results_)
		results.to_csv(to_save_cv)

	return net_cv

def multi_layer_network_performance(name, save = True):
	"""

	"""

	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	X_train, X_test = split.standardize(name, X_train, X_test)

	def multi_layer_network(neurons_l1,neurons_ol, num_layers, shape, learning_rate, 
	                        reg_l1, reg_ol, epochs, batch_size, X_train = X_train, y_train = y_train):
		"""

		"""
	
		net = Sequential()
		net.add(Dense(neurons_l1, activation='relu', input_shape= shape, kernel_regularizer=regularizers.l2(reg_l1)))
		for i in np.arange(num_layers):
			net.add(Dense(neurons_ol[i], activation='relu', input_shape= shape, 
			              kernel_regularizer=regularizers.l2(reg_ol[i])))
		net.add(Dense(1, activation='linear'))
		optimizer = Adam(learning_rate = learning_rate)
		net.compile(optimizer=optimizer, loss='mean_squared_error')
		net.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, workers=7, use_multiprocessing=True)
		return net

	results = pd.read_csv(Path().resolve().joinpath('models', 'cross_validation_outcomes', 
	                                                'neural_network', 
	                                                '{}_{}_{}.csv'.format(name, 'multi_layer_network', 'randomized')), index_col = 0)

	bestr2 = results.loc[results['rank_test_$R^2$'] == 1, 'params'].values[0]
	bestmae = results.loc[results['rank_test_Mean Absolute Error'] == 1, 'params'].values[0]
	bestrmse = results.loc[results['rank_test_Root Mean Square Error'] == 1, 'params'].values[0]

	display_name = ds.get_names()[name]
	dict_list = [bestr2,bestmae, bestrmse]
	unique_dict_list = [dict(t) for t in {tuple(sorted(d.items())) for d in dict_list}]
	performance = pd.DataFrame()
	for item in unique_dict_list:
		preds = multi_layer_network(**item).predict(X_test)
		performance = pd.concat([performance, metrics.apply_metrics('{} Single Layer NN'.format(display_name),
		 y_test, preds)], axis = 0)

	if save:
		to_save = Path().resolve().joinpath('models', 'neural_network',
		'{}.csv'.format('multi_layer_network'))
		performance.to_csv(to_save)

import numpy as np
import pandas as pd
from pathlib import Path
import src.data.datasets as ds
import src.data.train_test_split as split
from IPython.display import display, Image, Markdown

def main_datasets():
	'''

	'''

	pd.set_option('display.max_rows', 2)
	number_numerical = ds.get_number_numerical()
	dsets = ds.load_datasets(names = ['dataset_1', 'dataset_2', 'dataset_3'])
	display(Markdown('### `Dataset 1:` {} features: {} numerical, {} categorical, 1 response'.format(\
	                                                                                     len(dsets['dataset_1'].columns)-1, 
	                                                                                     number_numerical['dataset_1'], 
	                                                                                     len(dsets['dataset_1'].columns)\
	                                                                                     -1-number_numerical['dataset_1'])))
	display(dsets['dataset_1'])
	display(Markdown('---'))
	display(Markdown('### `Dataset 2:` {} features: {} numerical, {} categorical, 1 response'.format(\
	                                                                                     len(dsets['dataset_2'].columns)-1, 
	                                                                                     number_numerical['dataset_2'], 
	                                                                                     len(dsets['dataset_2'].columns)\
	                                                                                     -1-number_numerical['dataset_2'])))
	display(dsets['dataset_2'])
	display(Markdown('---'))
	display(Markdown('### `Dataset 3:` {} features: {} numerical, {} categorical, 1 response'.format(\
	                                                                                     len(dsets['dataset_3'].columns)-1, 
	                                                                                     number_numerical['dataset_3'], 
	                                                                                     len(dsets['dataset_3'].columns)\
	                                                                                     -1-number_numerical['dataset_3'])))
	display(dsets['dataset_3'])
	display(Markdown('---'))

def main_datasets_split():
	'''

	'''

	number_numerical = ds.get_number_numerical()
	pd.set_option('display.max_rows', 2)
	dsets = split.split_subsets(['dataset_1', 'dataset_2', 'dataset_3'])
	display(Markdown('### `Dataset 1 Training Set:` {} features: {} numerical, {} categorical, 1 response'.format(\
	                                                                                     len(dsets['dataset_1'][4].columns), 
	                                                                                     number_numerical['dataset_1'], 
	                                                                                     len(dsets['dataset_1'][4].columns)\
	                                                                                     -number_numerical['dataset_1'])))
	display(dsets['dataset_1'][4])
	display(Markdown('---'))
	display(Markdown('### `Dataset 2 Training Set:` {} features: {} numerical, {} categorical, 1 response'.format(\
	                                                                                     len(dsets['dataset_2'][4].columns), 
	                                                                                     number_numerical['dataset_2'], 
	                                                                                     len(dsets['dataset_2'][4].columns)\
	                                                                                     -number_numerical['dataset_2'])))
	display(dsets['dataset_2'][4])
	display(Markdown('---'))
	display(Markdown('### `Dataset 3 Training Set:` {} features: {} numerical, {} categorical, 1 response'.format(\
	                                                                                     len(dsets['dataset_3'][4].columns), 
	                                                                                     number_numerical['dataset_3'], 
	                                                                                     len(dsets['dataset_3'][4].columns)\
	                                                                                     -number_numerical['dataset_3'])))
	display(dsets['dataset_3'][4])
	display(Markdown('---'))

def plots():
	'''Plots all the different plotting functions in one call.

	'''

	display(Markdown('### `Dataset 1:`'))
	display(Image("visualizations/all_plots_dataset_1.png"))
	display(Markdown('---'))
	display(Markdown('### `Dataset 2:`'))
	display(Image("visualizations/all_plots_dataset_2.png"))
	display(Markdown('---'))
	display(Markdown('### `Dataset 3:`'))
	display(Image("visualizations/all_plots_dataset_3.png"))
	display(Markdown('---'))

def averages():
	'''

	'''

	pd.set_option('display.float_format', lambda x: '%.4f' % x)
	pd.set_option('display.max_rows', 3)
	averages = pd.read_csv(Path().resolve().joinpath('models', 'baseline', 'averages.csv'), 
		                      index_col = 0)

	display(averages)

def ols():
	'''

	'''

	pd.set_option('display.float_format', lambda x: '%.4f' % x)
	ols = pd.read_csv(Path().resolve().joinpath('models', 'baseline', 'OLS.csv'), 
		                      index_col = 0)

	display(ols)

def tests():
	'''

	'''

	pd.set_option('display.float_format', lambda x: '%.4f' % x)
	pd.set_option('display.max_rows', 8)
	d1 = pd.read_csv(Path().resolve().joinpath('features', 'statistical_tests', 'dataset_1.csv'), 
		                      index_col = 0)
	d2 = pd.read_csv(Path().resolve().joinpath('features', 'statistical_tests', 'dataset_2.csv'), 
		                      index_col = 0)
	d3 = pd.read_csv(Path().resolve().joinpath('features', 'statistical_tests', 'dataset_3.csv'), 
		                      index_col = 0)

	display(Markdown('### `Dataset 1:`'))
	display(d1)
	display(Markdown('### `Dataset 2:`'))
	display(d2)
	display(Markdown('### `Dataset 3:`'))
	display(d3)

def subsets():
	'''

	'''

	name_dict = ds.get_names()
	rf_dict = ds.get_removed_features()
	del name_dict['dataset_1']
	del name_dict['dataset_2']
	del name_dict['dataset_3']

	for file_name in list(name_dict.keys()):
		display(Markdown('### `{}` removed features:  {}'.\
		                 format(name_dict[file_name],rf_dict[file_name])))


def clustering():
	'''

	'''

	display(Markdown('### `Dataset 1:`'))
	display(Image("visualizations/clustering/dataset_1.png"))
	display(Markdown('---'))
	display(Markdown('### `Dataset 2:`'))
	display(Image("visualizations/clustering/dataset_2.png"))
	display(Markdown('---'))
	display(Markdown('### `Dataset 3:`'))
	display(Image("visualizations/clustering/dataset_3.png"))
	display(Markdown('---'))

def linear():
	'''

	'''

	df = pd.read_csv(Path().resolve().joinpath('models', 'linear', 'performance_outcomes_all.csv'), index_col = 0)
	r2 = df.loc[df['$OSR^2$']  == max(df['$OSR^2$'])].index.values
	r2 = ", ".join(str(x) for x in r2)
	# evs = df.loc[df['Explained Variance Score'] == \
	#                 max(df['Explained Variance Score'])].index.values
	# evs = ", ".join(str(x) for x in evs)
	mae = df.loc[df['Mean Absolute Error'] == \
	                min(df['Mean Absolute Error'])].index.values
	mae = ", ".join(str(x) for x in mae)
	rmse = df.loc[df['Root Mean Square Error'] == \
	                 min(df['Root Mean Square Error'])].index.values
	rmse = ", ".join(str(x) for x in rmse)
	# mape = df.loc[df['Mean Absolute Percent Error'] == \
	#                  min(df['Mean Absolute Percent Error'])].index.values
	# mape = ", ".join(str(x) for x in mape)

	display(Markdown('### $OSR^2$: {}: {}'.format(np.round(max(df['$OSR^2$']), 4), r2)))
	# display(Markdown('### Explained Variance Score: {}: {}'.format(np.round(max(df['Explained Variance Score']), 3), \
	#                                                                  evs)))
	display(Markdown('### Mean Absolute Error: {}: {}'.format(np.round(min(df['Mean Absolute Error']), 4), mae)))
	display(Markdown('### Root Mean Square Error: {}: {}'.format(np.round(min(df['Root Mean Square Error']),4), rmse)))
	# display(Markdown('### Mean Absolute Percent Error: {}: {}'.format(np.round(min(df['Mean Absolute Percent Error'])),\
	#                                                                     mape)))
	display(Markdown('---'))

def ensemble():
	'''

	'''

	df = pd.read_csv(Path().resolve().joinpath('models', 'ensemble', 'performance_outcomes_all.csv'), index_col = 0)
	r2 = df.loc[df['$R^2$']  == max(df['$R^2$'])].index.values
	r2 = ", ".join(str(x) for x in r2)
	# evs = df.loc[df['Explained Variance Score'] == \
	#                 max(df['Explained Variance Score'])].index.values
	# evs = ", ".join(str(x) for x in evs)
	mae = df.loc[df['Mean Absolute Error'] == \
	                min(df['Mean Absolute Error'])].index.values
	mae = ", ".join(str(x) for x in mae)
	rmse = df.loc[df['Root Mean Square Error'] == \
	                 min(df['Root Mean Square Error'])].index.values
	rmse = ", ".join(str(x) for x in rmse)
	# mape = df.loc[df['Mean Absolute Percent Error'] == \
	#                  min(df['Mean Absolute Percent Error'])].index.values
	# mape = ", ".join(str(x) for x in mape)

	display(Markdown('### $R^2$: {}: {}'.format(np.round(max(df['$R^2$']), 4), r2)))
	# display(Markdown('### Explained Variance Score: {}: {}'.format(np.round(max(df['Explained Variance Score']), 3), \
	#                                                                  evs)))
	display(Markdown('### Mean Absolute Error: {}: {}'.format(np.round(min(df['Mean Absolute Error']), 4), mae)))
	display(Markdown('### Root Mean Square Error: {}: {}'.format(np.round(min(df['Root Mean Square Error']),4), rmse)))
	# display(Markdown('### Mean Absolute Percent Error: {}: {}'.format(np.round(min(df['Mean Absolute Percent Error'])),\
	#                                                                     mape)))
	display(Markdown('---'))




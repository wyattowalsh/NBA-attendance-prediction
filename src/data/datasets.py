import numpy as np
import pandas as pd
import warnings
from pathlib import Path

def get_removed_features():
	'''Creates and returns dictionary of features removed in data subsets

	'''
	file_names = ['dataset_1_1', 'dataset_1_3',
	'dataset_2_1', 'dataset_2_3',
	"dataset_3_1", "dataset_3_2", "dataset_3_3"]

	removed_features = ['LS Win %, Capacity, Last Game',
	'Curr Win %, Last Attendance vs Opp',
	'LS Win %, Last Game, Capacity',
	'Curr Win %, Last Attendance vs Opp',
	'Curr Win %, LS Win %, Last Game, Capacity',
	'V Pop, Last Attendance vs Opp',
	'V Pop, H Pop, Curr Win %, Last Attendance vs Opp']

	rf_dict = dict(zip(file_names, removed_features))

	return rf_dict

def get_names():
	'''Creates and returns dictionary of display names for the different datasets' file names

	'''

	file_names = ['dataset_1','dataset_1_1', 'dataset_1_3',
	"dataset_2", 'dataset_2_1', 'dataset_2_3',
	"dataset_3", "dataset_3_1", "dataset_3_2", "dataset_3_3"]

	names = ['Dataset 1', "Dataset 1 Subset 1", "Dataset 1 Subset 2",
	'Dataset 2', "Dataset 2 Subset 1", "Dataset 2 Subset 2",
	"Dataset 3", "Dataset 3 Subset 1", "Dataset 3 Subset 2", "Dataset 3 Subset 3"]

	name_dict = dict(zip(file_names, names))

	return name_dict

def load_dataset(name): 
	'''returns dataframe from a local csv in the data/processed directory

	'''

	try: 
		dataset = pd.read_csv(Path().resolve().joinpath('data', 'processed', '{}.csv'.format(name)), 
							  index_col = 0)
		dataset.index = pd.to_datetime(dataset.index)
		return dataset
	except:
		warnings.warn('{} does not exist'.format(name))

def load_datasets(names = ['dataset_1','dataset_1_1', 'dataset_1_3',
				  "dataset_2", 'dataset_2_1', 'dataset_2_3',
				  "dataset_3", "dataset_3_1", "dataset_3_2", "dataset_3_3"]):
	'''returns dictionary where keys are the file names associated with the different datasets 
	and values are the associated dataframes 

	'''

	datasets = {}
	for name in names:
		datasets[name] = load_dataset(name)
		if datasets[name] is None:
			warnings.warn('{} does not exist'.format(name))
			return
	return datasets

def save_dataset(name, data):
	"""saves a processed dataset as csv in the data/processed directory

	"""

	data = data.copy()
	to_save = Path().resolve().joinpath('data', 'processed', '{}.csv'.format(name))
	data.to_csv(to_save)

def create_datasets():
	'''Creates datasets and subsets based off of raw data and saves them as .csv files in the data/processed directory

	'''

	dataset()
	dataset_1()
	dataset_1_1()
	dataset_1_3()
	dataset_2()
	dataset_2_1()
	dataset_2_3()
	dataset_3()
	dataset_3_1()
	dataset_3_2()
	dataset_3_3()
	dataset_4()

def get_number_numerical():
	'''Returns a dictionary of the number of numerical features each dataset has

	'''

	numerical_features = {'dataset_1': 5,
	'dataset_1_1': 2,
	'dataset_1_3': 3,
	'dataset_2': 5,
	'dataset_2_1': 2,
	'dataset_2_3': 3,
	'dataset_3': 7,
	'dataset_3_1': 3,
	'dataset_3_2': 5,
	'dataset_3_3': 3}
	
	return numerical_features


def dataset():
	"""This is the master dataset containing games from all the years scraped.

	It does not include popularity or capacity data.
	It does include the lagged attendance feature and last attendance versus the same opponent
	It also removes games with attendance over 25,000
	"""
	data = pd.read_csv(Path().resolve().joinpath('data', 'raw', 'game_data.csv'), index_col = 0)
	data = data.sort_values('Time').reset_index(drop=True).set_index('Time', drop=True)
	data.index = pd.to_datetime(data.index)
	# Remove extreme outliers
	data = data.loc[data['Attendance'] <= 25000]

	data['Last Game'] = np.nan
	data["Last Attendance vs Opp"] = np.nan
	teams = data['Home'].unique()
	for team in teams:
		data.loc[data['Home'] == team, 'Last Game'] = (data.loc[data['Home'] == team, 'Attendance'].shift(1)) 
		data.loc[data['Home'] == team] = data.loc[data['Home'] == team].sort_values(['Visitor', 'Time'])
		tot_scores = np.array([])
		for visitor in data.loc[data['Home'] == team]['Visitor'].unique():
			scores = data.loc[data['Home'] == team].loc[data.loc[data['Home'] == team]['Visitor'] == visitor]['Attendance'].values
			scores = np.append(np.array([0]), scores[0:len(scores)-1])
			tot_scores = np.append(tot_scores, scores)
		data.loc[data['Home'] == team, 'Last Attendance vs Opp'] = tot_scores
	data = data.dropna()

	# Remove seemingly redundant or irrelevant features
	data = data.drop(["V PTS", "H PTS", "Match-up"], axis=1)
	data = data[['V Pop', 'H Pop', 'Curr Win %',  'LS Win %','Last Game', 'Last Attendance vs Opp', 
	'Capacity', "Home", "Visitor",'Playoffs?', 'Last Five', 'Day of Week','Month', 'Rivalry?', 'Attendance']]
	data = data.dropna()
	data = data.loc[data.index < pd.to_datetime('2019-12')]
	save_dataset('dataset', data)
	return


def dataset_1():
	'''This is the first cleaned dataset.
	
	It is from 1999 to 12/2019, does not contain popularity data or rivalry data,
	but does contain all other features found in the master dataset
	'''

	data = load_dataset('dataset')
	data = data.drop(['V Pop', "H Pop", 'Rivalry?'], axis = 1)
	data = data.loc[data.index.year >= 1999]
	save_dataset('dataset_1', data)
	return

def dataset_1_1():
	'''This is a subset of dataset_1 based off of VIF analysis
	

	'''

	data = load_dataset('dataset_1')
	data = data.drop(['LS Win %', 'Capacity', 'Last Game'], axis = 1)
	save_dataset('dataset_1_1', data)

def dataset_1_3():
	'''This is a subset of dataset_1 based off of estimated mutual information analysis
	
	Features dropped are: Curr Win % and Last Attendance vs Opp
	'''

	data = load_dataset('dataset_1')
	data = data.drop(['Curr Win %', 'Last Attendance vs Opp'], axis = 1)
	save_dataset('dataset_1_3', data)


def dataset_2():
	'''This is the second cleaned dataset
	
	It is from 1999 to 12/2019
	Removes games played in old stadiums from the master dataset
	Does not include popularity or rival data
	''' 

	data = load_dataset('dataset')
	stadiums = pd.read_csv(Path().resolve().joinpath('data', 'raw', 'stadiums_data.csv'), index_col = 0)

	# Remove games that were played in old stadiums
	teams = data['Home'].unique()
	for team in teams:
		earliest = pd.to_numeric(stadiums.loc[stadiums['Home'] == team]['Opened'].values[0])
		data.loc[data['Home'] == team] = data.loc[data['Home'] == team].loc[data.loc[data['Home'] == team].index >= pd.Timestamp(earliest, 7, 1)]

	data = data.drop(["V Pop", "H Pop",'Rivalry?'], axis = 1)
	data = data.dropna()
	data = data.loc[data.index.year >= 1999]
	save_dataset('dataset_2', data)
	return 

def dataset_2_1():
	'''This is a subset of dataset_2 based off of VIF analysis 

	Features dropped are: LS Win %, Last Game, and Last Attendance vs Opp
	'''

	data = load_dataset('dataset_2')
	data = data.drop(['LS Win %', 'Last Game', 'Capacity'], axis = 1)
	save_dataset('dataset_2_1', data)


def dataset_2_3():
	'''This is a subset of dataset_2 based off of estimated mutual information analysis
	
	Features dropped are: Curr Win % and Last Attendance vs Opp
	'''

	data = load_dataset('dataset_2')
	data = data.drop(['Curr Win %', 'Last Attendance vs Opp'], axis = 1)
	save_dataset('dataset_2_3', data)

def dataset_3():
	'''This is the third cleaned dataset
	
	It is from 1/2004 to 12/2019
	It removes games that were played in old stadiums
	It does not include rivalry data
	'''

	data = load_dataset('dataset')
	stadiums = pd.read_csv(Path().resolve().joinpath('data', 'raw', 'stadiums_data.csv'), index_col = 0)

	# Remove games that were played in old stadiums
	teams = data['Home'].unique()
	for team in teams:
		earliest = pd.to_numeric(stadiums.loc[stadiums['Home'] == team]['Opened'].values[0])
		data.loc[data['Home'] == team] = data.loc[data['Home'] == team].loc[data.loc[data['Home'] == team].index >= pd.Timestamp(earliest, 7, 1)]

	data = data.drop(['Rivalry?'], axis = 1)
	data = data.dropna()
	data = data.loc[data.index.year >= 2004]    
	save_dataset('dataset_3', data)

def dataset_3_1():
	'''This is a subset of dataset_3 based off of VIF analysis 

	Features dropped are: Capacity, Curr Win %, LS Win %, and Last Game
	'''

	data = load_dataset('dataset_3')
	data = data.drop(['Capacity', 'Curr Win %', 'LS Win %', 'Last Game'], axis = 1)
	save_dataset("dataset_3_1", data)

def dataset_3_2():
	'''This is a subset of dataset_3 based off of F-test analysis
	
	Features dropped are: Last Attendance vs Opp and V Pop
	'''

	data = load_dataset('dataset_3')
	data = data.drop(['Last Attendance vs Opp', 'V Pop'], axis = 1)
	save_dataset("dataset_3_2", data)

def dataset_3_3():
	'''This is a subset of dataset_3 based off of estimated mutual information analysis
	
	Features dropped are: Curr Win %, V Pop, H Pop, and Last Attendance vs Opp
	'''

	data = load_dataset('dataset_3')
	data = data.drop(['Curr Win %', 'V Pop', 'H Pop', 'Last Attendance vs Opp'], axis = 1)
	save_dataset("dataset_3_3", data)

def dataset_4():
	'''

	'''

	data = load_dataset('dataset')
	data = data.drop(['Rivalry?'], axis = 1)
	data = data.dropna()
	data = data.loc[data.index.year >= 2004]    
	save_dataset('dataset_4', data)

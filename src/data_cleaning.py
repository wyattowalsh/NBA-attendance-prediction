"""Module for cleaning collected data"""

import os
import numpy as np
import pandas as pd


def clean_season(path_to_raw_file):
	"""Processes raw seasonal data
	Groups by game and adds columns for second team

	inputs:
		path_to_raw_file - string file path

	outputs:
		processed seasonal dataset

	"""
	df = pd.read_csv(path_to_raw_file, dtype= {'GAME_ID': str})
	df['GAME_ID'] = df["GAME_ID"].apply(str)
	df = df.groupby(['GAME_ID', 'SEASON_ID', 'GAME_DATE']).agg(list)
	columns_old = df.columns
	columns_new = sum([[x + '_1'] + [x + '_2'] for x in df.columns], [])
	df = df.reset_index()
	cleaned = pd.concat([df.iloc[:, 0:3]] + [pd.DataFrame(df[x].to_list(), \
                        columns = [columns_new[i*2], columns_new[i*2+1]]) for i, x in enumerate(columns_old)], 1)
	save_path = path_to_raw_file.replace('raw', 'processed')
	if not os.path.exists(save_path):
		cleaned.to_csv(save_path, index=False)
	return cleaned

def clean_seasons():
	"""Processes all seasonal data within the data/raw/seasons directory
	Saves processed data within data/cleaned/seasons directory if not already existing

	inputs: 
		None

	outputs:
		None

	"""
	raw_seasons_directory =  'data/raw/seasons/'
	for file in os.listdir(raw_seasons_directory):
		if '.csv' in file:
			clean_season(raw_seasons_directory + file)
			print(file, ' successfully processed')
'''Collects and updates dataset'''


import os
import time
import numpy as np
import pandas as pd
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import leaguegamelog


def get_teams():
	"""Queries all teams from NBA stats
	Saves dataframe as csv if /data/raw/teams.csv does not exist

	inputs:
		None

	outputs:
		Pandas dataframe of all NBA teams
	"""
	teams_df = pd.DataFrame(teams.get_teams())
	if not os.path.exists('data/raw/teams.csv'):
		teams_df.to_csv('data/raw/teams.csv')
	return teams_df

def get_players():
	"""Queries all players from NBA stats
	Saves dataframe as .csv if /data/raw/players.csv does not exist

	inputs:
		None

	outputs:
		Pandas dataframe of all NBA players
	"""
	players_df = pd.DataFrame(players.get_players())
	if not os.path.exists('data/raw/players.csv'):
		players_df.to_csv('data/raw/players.csv', index=False)
	return players_df

def get_seasons(year_start=1946, year_end=2021, timeout=120, rest_time=10):
	"""Queries all games by season 1983-2020 from NBA stats
	Saves dataframe as .csv if /data/raw/games.csv does not exit

	inputs:
		year_start - integer for starting year of first season to collect
		year_end - integer for ending year of last season to collect
		timeout - timeout amount for queries 

	outputs:
		Pandas dataframe of all NBA games across all seasons (1946-2020)
	"""
	seasons = [str(x) + '-' + str(x+1)[-2:] for x in list(range(year_start, year_end))]
	for season in seasons:
		try:
			addition = leaguegamelog.LeagueGameLog(season=season, timeout=timeout).get_data_frames()[0]
			time.sleep(rest_time)
			if not os.path.exists('data/raw/seasons/{}.csv'.format(season)):
				addition.to_csv('data/raw/seasons/{}.csv'.format(season), index=False)
		except:
			print('Failed for season: ', season)
			continue

def get_games_season(season_data_file_path):
	"""create"""















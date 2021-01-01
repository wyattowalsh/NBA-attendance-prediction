import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

def scrape_and_compile(start_year=1989, end_year=2021,
	ref="https://www.basketball-reference.com/leagues/NBA_{}_games-{}.html"):
	'''Scrapes basketball game data over a given time period, saves as CSV, and returns a dataframe.

	Scrapes data from basketball-reference.com (unless inputted otherwise) from start_year to end_year-1.
	Start_year's default value is 1989 since the oldest stadium opened in 1990 (accounting for lag year).
	After scraping from the site, conducts feature engineering and dataframe rearrangement/reformatting.
	'''

	# Declare season months, rivalries
	months = ['october', 'november', 'december', 'january','february', 'march', 'april', 'may', 'june']

	# Rivalries from: https://en.wikipedia.org/wiki/List_of_National_Basketball_Association_rivalries
	rivalries = {frozenset({'Cleveland Cavaliers', 'Golden State Warriors'}),
				frozenset({'Boston Celtics', 'Los Angeles Lakers'}),
				frozenset({'Detroit Pistons', 'Los Angeles Lakers'}),
				frozenset({'Philadelphia 76ers', 'Boston Celtics'}),
				frozenset({'Boston Celtics', 'New York Knicks'}),
				frozenset({'New York Knicks', 'Brooklyn Nets'}),
				frozenset({'Chicago Bulls', 'Detroit Pistons'}),
				frozenset({'Chicago Bulls', 'Cleveland Cavaliers'}),
				frozenset({'Miami Heat', 'Orlando Magic'}),
				frozenset({'Chicago Bulls', 'Miami Heat'}),
				frozenset({'Chicago Bulls', 'New York Knicks'}),
				frozenset({'Boston Celtics', 'Detroit Pistons'}),
				frozenset({'Miami Heat', 'New York Knicks'}),
				frozenset({'Indiana Pacers', 'New York Knicks'}),
				frozenset({'Los Angeles Clippers', 'Los Angeles Lakers'}),
				frozenset({'Dallas Mavericks', 'Houston Rockets'}),
				frozenset({'Houston Rockets', 'San Antonio Spurs'}),
				frozenset({'Houston Rockets', 'Utah Jazz'}),
				frozenset({'San Antonio Spurs', 'Los Angeles Lakers'}),
				frozenset({'Phoenix Suns', 'San Antonio Spurs'})}

	pop_yr_df = pd.read_csv(Path().resolve().joinpath('data', 'raw', 'popularity_data.csv'), index_col = 0)
	pop_yr_df.columns = pd.to_datetime(pop_yr_df.columns)
	all_data_df = pd.DataFrame()
	for year in np.arange(start_year, end_year):
		year_data_df = pd.DataFrame()
		for month in months:
			# For each month in the season for a given year, scrape all the listed games
			site = requests.get(ref.format(year, month))
			soup = BeautifulSoup(site.content, 'html.parser')
			sched = soup.find(id='schedule')
			try:
				tr = sched.find_all('tr')
			except:
				continue
			data = [[th.getText() for th in tr[i].findAll('th')] for i in range(len(tr))]
			col_labels = data[0]
			dates = np.array(data[1:])
			dates_df = pd.DataFrame(data=dates, columns=['Date'])
			data = [[td.getText() for td in tr[i].findAll('td')] for i in range(len(tr))]
			data_df = pd.DataFrame(data=data[1:], columns=col_labels[1:])
			both_df = pd.concat([dates_df, data_df], axis=1)
			year_data_df = year_data_df.append(both_df, ignore_index=True)

		# Combine date and time (Games before 2000 seem to not have time component)
		try:
			year_data_df['Time'] = year_data_df["Date"].map(str) + " " + year_data_df["Start (ET)"].map(str)
		except:
			year_data_df['Time'] = year_data_df["Date"].map(str)

		# Determine row at which playoffs begin, drop that row, and add column to show if row is playoff game
		try:
			playoffs = year_data_df.loc[year_data_df['Date'] == "Playoffs"].index.values[0]
			year_data_df = year_data_df.drop(year_data_df.index[playoffs]).reset_index(drop=True)
			year_data_df['Playoffs?'] = np.where(year_data_df.index.values < playoffs, 0, 1)
		except:
			year_data_df['Playoffs?'] = 0

		# Drop rows where nan for attendance
		year_data_df = year_data_df[year_data_df['Attend.']!= np.nan].reset_index(drop=True)

		# Drop rows where note exists since usually saying game was held at different stadium
		year_data_df = year_data_df[year_data_df.Notes ==''].reset_index(drop=True)

		# Drop unnecessary columns
		try:
			year_data_df = year_data_df.drop(['Date', 'Start (ET)', '\xa0', '\xa0', 'Notes'], axis=1)
		except:
			year_data_df = year_data_df.drop(['Date', '\xa0', '\xa0', 'Notes'], axis=1)

		# Rename and rearrange columns
		year_data_df.columns = ['Visitor', 'V PTS', 'Home','H PTS', 'Attendance', 'Time', 'Playoffs?']
		cols = ['Attendance', 'Time', 'Visitor','V PTS', 'Home', 'H PTS', 'Playoffs?']
		year_data_df = year_data_df.loc[:, cols]

		# Filter to existing teams
		year_data_df = year_data_df.loc[year_data_df['Home'].isin(pop_yr_df.index.values)]
		year_data_df = year_data_df.loc[year_data_df['Visitor'].isin(pop_yr_df.index.values)]

		# Apply types to each columns
		year_data_df['Attendance'] = pd.to_numeric(year_data_df.loc[:, 'Attendance'].str.replace(',', ''))
		year_data_df['Time'] = pd.to_datetime(year_data_df['Time'].values, infer_datetime_format=True)
		year_data_df['V PTS'] = pd.to_numeric(year_data_df.loc[:, 'V PTS'].values)
		year_data_df["H PTS"] = pd.to_numeric(year_data_df.loc[:, "H PTS"].values)

		# -Add additional columns of data-

		# Popularity of teams
		year_data_df['V Pop'] = 0
		year_data_df['H Pop'] = 0
		if year in pop_yr_df.columns.year.unique()[1:]:
			for month in year_data_df['Time'].dt.month.unique():
				# If first half of season
				if month < 13 and month > 8:
					if year == 2019 and month == 12:
						continue
					for team in pop_yr_df.index.values:
						pop_data = pop_yr_df[str(year-1)+'-' + str(month) + '-01']
						pop_data = pop_data.loc[pop_data.index == team].values[0]
						year_data_df.loc[(year_data_df['Time'].dt.month == month) & 
										(year_data_df['Visitor'] == team), "V Pop"] = pop_data
						year_data_df.loc[(year_data_df['Time'].dt.month == month) &
										(year_data_df['Home'] == team), "H Pop"] = pop_data
				# If second half of season
				else:
					for team in pop_yr_df.index.values:
						pop_data = pop_yr_df[str(year)+'-' + str(month) + '-01']
						pop_data = pop_data.loc[pop_data.index == team].values[0]
						year_data_df.loc[(year_data_df['Time'].dt.month == month) &
										(year_data_df['Visitor'] == team), "V Pop"] = pop_data
						year_data_df.loc[(year_data_df['Time'].dt.month == month) &
										(year_data_df['Home'] == team), "H Pop"] = pop_data
		elif year == 2020:
			for month in year_data_df['Time'].dt.month.unique():
				for team in pop_yr_df.index.values:
					if month == 12:
						continue
					pop_data = pop_yr_df[str(year-1)+'-' + str(month) + '-01']
					pop_data = pop_data.loc[pop_data.index == team].values[0]
					year_data_df.loc[(year_data_df['Time'].dt.month == month) & 
									(year_data_df['Visitor'] == team), "V Pop"] = pop_data
					year_data_df.loc[(year_data_df['Time'].dt.month == month) &
									(year_data_df['Home'] == team), "H Pop"] = pop_data
		else:
			for month in year_data_df['Time'].dt.month.unique():
				for team in pop_yr_df.index.values:
					pop_data = pop_yr_df[str(2004)+'-' + str(month) + '-01']
					pop_data = pop_data.loc[pop_data.index == team].values[0]
					year_data_df.loc[(year_data_df['Time'].dt.month == month) &
									(year_data_df['Visitor'] == team), "V Pop"] = pop_data
					year_data_df.loc[(year_data_df['Time'].dt.month == month) &
									(year_data_df['Home'] == team), "H Pop"] = pop_data
									
		# From point data, add last five game record for each game
		year_data_df['Last Five'] = 0
		year_data_df['Capacity'] = np.nan
		for team in pop_yr_df.index.values:
			year_data_df['Win?'] = 0
			year_data_df.loc[((year_data_df["Home"] == team) & (year_data_df["V PTS"] < year_data_df['H PTS'])) |
			((year_data_df["Visitor"] == team) & (year_data_df["V PTS"] > year_data_df['H PTS'])), "Win?".format(team)] = 1
			year_data_df.loc[(year_data_df["Home"] == team) | (year_data_df["Visitor"] == team), 'Curr Win %'] = \
			year_data_df.loc[(year_data_df["Home"] == team) | 
			(year_data_df["Visitor"] == team), "Win?".format(team)].expanding().mean()
			year_data_df.loc[(year_data_df["Home"] == team) | (year_data_df["Visitor"] == team), 'Last Five'] = \
			year_data_df.loc[(year_data_df["Home"] == team) | (year_data_df["Visitor"] == team), "Win?"].\
			rolling(5, min_periods=1).sum()
			year_data_df = year_data_df.drop(["Win?"], axis=1)
			try:
				year_data_df.loc[year_data_df['Home'] == team, 'Capacity'] = \
				max(year_data_df.loc[year_data_df['Home'] == team, "Attendance"].values)
			except:
				continue
			 
		# Day of the week game was on and month
		year_data_df['Day of Week'] = year_data_df['Time'].dt.day_name()
		year_data_df['Month'] = year_data_df['Time'].dt.month_name()

		# If game is rivalry
		year_data_df['Match-up'] = list(map(frozenset,zip(year_data_df['Visitor'], year_data_df['Home'])))
		year_data_df['Rivalry?'] = np.where(year_data_df['Match-up'].isin(rivalries), 1, 0)

		# Append current year's data to past years
		all_data_df = all_data_df.append(year_data_df, ignore_index=True)

	all_data_df = all_data_df.sort_values('Time')

	all_data_df['LS Win %'] = np.nan
	for year in all_data_df["Time"].dt.year.unique()[1:]:
		for team in pop_yr_df.index.values:
			try:
				win_p = all_data_df.loc[(all_data_df['Time'].dt.year == year) & 
										((all_data_df["Home"] == team) | (all_data_df["Visitor"] == team)) &
										(all_data_df["Time"].dt.month < 8), "Curr Win %"].values[-1]
				win_p_ls = all_data_df.loc[(all_data_df['Time'].dt.year == year-1) &
											((all_data_df["Home"] == team) | (all_data_df["Visitor"] == team)) &
											(all_data_df["Time"].dt.month < 8), "Curr Win %"].values[-1]

				all_data_df.loc[(all_data_df['Time'].dt.year == year) & (all_data_df["Home"] == team) &
				(all_data_df["Time"].dt.month > 8), "LS Win %"] = win_p

				all_data_df.loc[(all_data_df['Time'].dt.year == year) & (all_data_df["Home"] == team) &
				(all_data_df["Time"].dt.month < 8), "LS Win %"] = win_p_ls
			except:
				pass
	all_data_df = all_data_df.dropna()
	to_save = Path().resolve().joinpath('data', 'raw', 'game_data.csv')
	all_data_df.to_csv(to_save)
	return all_data_df

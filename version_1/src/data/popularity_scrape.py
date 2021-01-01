import pandas as pd
from pathlib import Path
from pytrends.request import TrendReq

def scrape():
	'''Uses pytrends API (https://github.com/GeneralMills/pytrends) to collect Google Trends Data.

	Google Trends only permits 5 search items per query and can temporarily block users.
	Earliest data available is from 2004. 
	'''

	pytrends = TrendReq(hl='en-US', retries=2, backoff_factor=0.1)
	wiki = pd.read_csv(Path().resolve().joinpath('data', 'raw', 'stadiums_data.csv'), index_col = 0)
	teams_list = wiki["Home"].values.tolist()
	teams_list = [teams_list[i:i + 5] for i in range(0, len(teams_list), 5)]
	pop = pd.DataFrame()
	for teams in teams_list:
		pytrends.build_payload(teams, cat=0, timeframe='all')
		pop_5 = pytrends.interest_over_time()
		pop_5 = pop_5.drop(['isPartial'], axis=1)
		pop = pd.concat([pop, pop_5], axis=1)
	pop_yr_df = pop.transpose()
	to_save = Path().resolve().joinpath('data', 'raw', 'popularity_data.csv')
	pop_yr_df.to_csv(to_save)
	return pop_yr_df
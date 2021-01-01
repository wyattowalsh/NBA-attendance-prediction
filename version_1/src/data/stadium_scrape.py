import pandas as pd
from pathlib import Path

def scrape():
	'''Uses default pandas methods to scrape two wiki tables and compile them together/reformat.

	State abbreviations from: https://gist.github.com/rogerallen/1583593 .
	Abbreviations/geo-codes were created in case Google Trend data wants to be scraped by region.
	'''

	site = "https://en.wikipedia.org/wiki/National_Basketball_Association#Teams"
	wiki = pd.read_html(site, header=0, attrs={
					   'class': 'wikitable', 'style': 'width:100%;'})[0]
	wiki = wiki[['Team', "City, State", "Arena"]].loc[wiki['Team'] != wiki["City, State"]].\
		sort_values("Team").reset_index(drop=True)
	wiki = wiki.rename({"City, State": 'Geo', "Team": "Home"}, axis=1)
	wiki['Geo'] = wiki.Geo.str.split(", ").str[1]
	us_state_abbrev = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO',
					   'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA',
					   'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
					   'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA',
					   'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
					   'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
					   'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Northern Mariana Islands': 'MP',
					   'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Palau': 'PW', 'Pennsylsvania': 'PA', 'Puerto Rico': 'PR',
					   'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
					   'Vermont': 'VT', 'Virgin Islands': 'VI', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
					   'Wisconsin': 'WI', 'Wyoming': 'WY'}
	wiki = wiki.replace({"Geo": us_state_abbrev})
	wiki['Geo'] = 'US-' + wiki['Geo'].astype(str)
	site = 'https://en.wikipedia.org/wiki/List_of_National_Basketball_Association_arenas'
	stadiums = pd.read_html(site, header=0)[0]
	stadiums = stadiums[['Team(s)', "Capacity", "Opened"]]
	stadiums.columns = ['Home', "Capacity", "Opened"]
	# Madison Square Garden was reconstructed in 2013
	stadiums.at[13, 'Opened'] = 2013
	stadiums.sort_values("Home").reset_index(drop=True)
	wiki = wiki.merge(stadiums, on="Home")
	to_save = Path().resolve().joinpath('data', 'raw', 'stadiums_data.csv')
	wiki.to_csv(to_save)
	return wiki
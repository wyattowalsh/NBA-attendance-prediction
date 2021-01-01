import src.data.datasets as ds
import src.data.train_test_split as split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path


def create_all_plots(name, train):
	'''Plots all the different plotting functions in one call.

	'''
	train = train.copy()
	create_attendance_histogram(name, train)
	create_daily_histogram(name, train)
	create_daily_barchart(name, train)
	create_monthly_histogram(name, train)
	create_monthly_barchart(name, train)
	create_yearly_histogram(name, train)
	create_yearly_barchart(name, train)
	create_playoffs_histograms(name, train)
	create_win_percent_histograms(name, train)
	create_last_five_record_histograms(name, train)
	create_last_five_record_barchart(name, train)
	create_heatmap(name, train)

def create_plots(name, save = False):
	'''

	'''


	dset = ds.load_dataset(name)
	train = dset
	sns.set_style("whitegrid")
	sns.set_palette(sns.color_palette('bright',12))

	fig = plt.figure(figsize=(40, 27))
	gs = gridspec.GridSpec(4, 4)
	ax00 = plt.subplot(gs[0,0])
	ax01 = plt.subplot(gs[0,1])
	ax02 = plt.subplot(gs[0,2])
	ax03 = plt.subplot(gs[0,3])
	ax10= plt.subplot(gs[1,0])
	ax11= plt.subplot(gs[1,1])
	ax12 = plt.subplot(gs[1,2])
	ax13 = plt.subplot(gs[1,3])
	ax20 = plt.subplot(gs[2,0])
	ax21 = plt.subplot(gs[2,1])
	ax22 = plt.subplot(gs[2,2])
	ax23 = plt.subplot(gs[2,3])
	ax30 = plt.subplot(gs[3,0:2])
	ax31 = plt.subplot(gs[3,2:4])


	bins = np.arange(4000, 26001, 500)
	# fig, ax = plt.subplots(nrows = 4, ncols = 4, figsize=(40, 20))
	sns.distplot(train['Attendance'], ax=ax00, kde=False, norm_hist=True, bins = bins)
	ax00.set_xlabel('Attendance (# of people)', fontsize = 20)
	ax00.set_ylabel('Percent per person', fontsize = 20) 
	ax00.tick_params(labelsize=15)
	ax00.set_title(label = "Overall Attendance", fontsize = 25)

	days = train['Day of Week'].unique()
	for day in days:
		sns.distplot(train.loc[train['Day of Week'] == day]['Attendance'],
							ax = ax01, kde= False, norm_hist= True, bins = bins)
	ax01.set_title(label = "Attendance per Day", fontsize = 25)
	ax01.set_xlabel('Attendance (# of people)', fontsize = 20)
	ax01.set_ylabel('Percent per person', fontsize = 20) 
	ax01.tick_params(labelsize=15)
	ax01.legend(days,loc="upper right", fontsize=15)

	grouped = train[['Day of Week', 'Attendance']].groupby('Day of Week').mean()
	order = ['Monday', "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
	sns.barplot(x='Attendance', y=grouped.index, data=grouped, order=order, ci=None, orient='h', 
	             saturation=1, ax=ax20, palette = sns.color_palette("cubehelix", 7))
	ax20.set_xlim(16500,19000)
	ax20.set_xticks(range(16500,19001,500))
	ax20.set_xlabel('Average Attendance (# of people)', fontsize = 20)
	ax20.set_ylabel('Day of the Week', fontsize = 20) 
	ax20.tick_params(labelsize=15)
	ax20.set_title(label = "Average Attendance per Day", fontsize = 25)

	months = train['Month'].unique()
	for month in months:
		sns.distplot(train.loc[train['Month'] == month]['Attendance'],
							ax = ax02, kde= False, norm_hist= True, bins = bins)
	ax02.set_title(label = "Attendance per Month", fontsize = 25)
	ax02.set_xlabel('Attendance (# of people)', fontsize = 20)
	ax02.set_ylabel('Percent per person', fontsize = 20) 
	ax02.tick_params(labelsize=15)
	ax02.legend(months,loc="upper right", fontsize=15)

	grouped = train[['Month', 'Attendance']].groupby('Month').mean()
	order = ['October', 'November', 'December', 'January','February', 'March', 'April', 'May', 'June']
	sns.barplot(x='Attendance', y=grouped.index, data=grouped, order=order, ci=None, orient='h', 
	             saturation=1, ax=ax21, palette = sns.color_palette("cubehelix", 9))
	ax21.set_xlim(16500,20000)
	ax21.set_xticks(range(16500,20001,500))
	ax21.set_xlabel('Average Attendance (# of people)', fontsize = 20)
	ax21.set_ylabel('Month', fontsize = 20) 
	ax21.tick_params(labelsize=15)
	ax21.set_title(label = "Average Attendance per Month", fontsize = 25)

	train['Year'] = train.index.year
	grouped = train[['Year','Attendance']].groupby('Year').mean()
	order = np.sort(train.index.year.unique())
	sns.barplot(x='Attendance', y=grouped.index, data=grouped, order=order, ci=None, orient='h', 
	             saturation=1, ax=ax22, palette = sns.color_palette("cubehelix", 13))
	ax22.set_xlim(16500,18500)
	ax22.set_xticks(range(16500,18501,500))
	ax22.set_xlabel('Average Attendance (# of people)', fontsize = 20)
	ax22.set_ylabel('Year', fontsize = 20) 
	ax22.tick_params(labelsize=15)
	ax22.set_title(label = "Average Attendance per Year", fontsize = 25)


	years = np.arange(max(train['Year'].values) - 4, max(train['Year'].values)+1)
	for year in years:
		sns.distplot(train.loc[train.index.year == year]['Attendance'],
							ax = ax03, kde= False, norm_hist= True, bins = bins)
	ax03.set_title(label = "Attendance per Year", fontsize = 25)
	ax03.set_xlabel('Attendance (# of people)', fontsize = 20)
	ax03.set_ylabel('Percent per person', fontsize = 20) 
	ax03.tick_params(labelsize=15)
	ax03.legend(years,loc="upper right", fontsize=20)


	for j in [0,1]:
		sns.distplot(train.loc[train['Playoffs?'] == j]['Attendance'],
							ax = ax10, kde= False, norm_hist= True, bins = bins)
	ax10.set_title(label = "Attendance for Regular and Playoff Games", fontsize = 25)
	ax10.set_xlabel('Attendance (# of people)', fontsize = 20)
	ax10.set_ylabel('Percent per person', fontsize = 20) 
	ax10.tick_params(labelsize=15)
	ax10.legend(['Regular Season', 'Playoffs'],loc="upper right", fontsize=15)

	win_percent = np.arange(0,1,0.10)
	for i in win_percent:
		sns.distplot(train.loc[(train['Curr Win %'] >= i) & (train['Curr Win %'] < i+0.1)]['Attendance'],
                        ax=ax11, kde=False, norm_hist=True, bins = bins)
	ax11.set_title(label = "Attendance for Different Win Percentages", fontsize = 25)
	ax11.set_xlabel('Attendance (# of people)', fontsize = 20)
	ax11.set_ylabel('Percent per person', fontsize = 20) 
	ax11.tick_params(labelsize=15)
	ax11.legend(['0 %', '10 %', '20 %', '30 %', '40 %','50 %','60 %','70 %','80 %','90 %','100 %'], 
	                loc="upper right", fontsize=17)

	last_five = np.arange(0,6)
	for i in last_five:
		sns.distplot(train.loc[train['Last Five'] == i]['Attendance'],
                        ax=ax12, kde=False, norm_hist=True, bins = bins)
	ax12.set_title(label = "Attendance by Last Five Record", fontsize = 25)
	ax12.set_xlabel('Attendance (# of people)', fontsize = 20)
	ax12.set_ylabel('Percent per person', fontsize = 20) 
	ax12.tick_params(labelsize=15)
	ax12.legend(['0 Wins', '1 Win', '2 Wins', "3 Wins", "4 Wins", "5 Wins"],loc="upper right", fontsize=20)

	grouped = train[['Last Five','Attendance']].groupby('Last Five').mean()
	order = np.arange(0,6)
	sns.barplot(x='Attendance', y=grouped.index, data=grouped, order=order, ci=None, orient='h', 
	             saturation=1, ax=ax31, palette = sns.color_palette("cubehelix", 5))
	ax31.set_xlim(16500,18500)
	ax31.set_xticks(range(16500,18501,500))
	ax31.set_xlabel('Average Attendance (# of people)', fontsize = 20)
	ax31.set_ylabel('Record Over Last Five Games', fontsize = 20) 
	ax31.tick_params(labelsize=15)
	ax31.set_title(label = "Average Attendance per Last Five Record", fontsize = 25)
	
	num_numerical = ds.get_number_numerical()[name]
	train_num = train.iloc[:,0:num_numerical]
	heat = sns.heatmap(train_num.corr(),annot = True, ax = ax13, fmt = '.2f', 
	                   cbar = True, square = True, xticklabels= True, yticklabels = True,
	                  annot_kws={'size':16}, cmap = 'coolwarm', center= 0, vmin=-1, vmax=1,
	                  cbar_kws={"shrink": 1})
	ax13.set_title('Heatmap of Numerical Variable Correlation', size=25) 
	ax13.set_xticklabels(ax13.xaxis.get_majorticklabels(), rotation=60, size = 15)
	ax13.set_yticklabels(ax13.yaxis.get_majorticklabels(), rotation=0, size = 15)
	ax13.collections[0].colorbar.ax.tick_params(labelsize=15)

	# Make annotations larger if abs(correlation) above 0.2
	num_corrs = len(np.unique(train_num.corr().values.flatten()))
	bigs = []
	for i in np.arange(2,num_corrs+1):
	    val = round(np.sort(np.abs(np.unique(train_num.corr().values.flatten())))[-i],2)
	    if val > 0.2:
	        bigs = np.append(bigs, val)
	for text in heat.texts:
	    num =  pd.to_numeric(text.get_text())
	    i = np.where(bigs == abs(num))[0]
	    if i.size > 0:
	        text.set_color('white')
	        text.set_size(27-(i[0]*3))

	train.loc[train['Playoffs?'] == 0, "Playoffs?"] = 'Regular Season'
	train.loc[train['Playoffs?'] == 1, "Playoffs?"] = 'Playoffs'
	grouped = train[['Playoffs?','Attendance']].groupby('Playoffs?').mean()
	order = ['Regular Season', 'Playoffs']
	sns.barplot(x='Attendance', y=grouped.index, data=grouped, order=order, ci=None, orient='h', 
	             saturation=1, ax=ax23, palette = sns.color_palette("cubehelix", 5))
	ax23.set_xlim(16500,19500)
	ax23.set_xticks(range(16500,19501,500))
	ax23.set_xlabel('Average Attendance (# of people)', fontsize = 20)
	ax23.set_ylabel('Game Type', fontsize = 20) 
	ax23.tick_params(labelsize=15)
	ax23.set_title(label = "Average Attendance per Game Type", fontsize = 25)

	train[['Curr Win %']] = np.round(train[['Curr Win %']],1) * 100
	grouped = train[['Curr Win %','Attendance']].groupby('Curr Win %').mean()
	order = np.arange(0,101,10)
	sns.barplot(x='Attendance', y=grouped.index, data=grouped, order=order, ci=None, orient='h', 
	             saturation=1, ax=ax30, palette = sns.color_palette("cubehelix", 5))
	ax30.set_xlim(16500,19500)
	ax30.set_xticks(range(16500,19501,500))
	ax30.set_xlabel('Average Attendance (# of people)', fontsize = 20)
	ax30.set_ylabel('Current Win %', fontsize = 20) 
	ax30.tick_params(labelsize=15)
	ax30.set_title(label = "Average Attendance per Current Win %", fontsize = 25)

	# ax[3][2] = sns.pairplot(data = train_num)
	# ax[3][2].set_title('Pairplot of Numerical Variable Correlation', size=25) 
	# ax[3][2].set_xticklabels(ax[3][2].xaxis.get_majorticklabels(), rotation=60, size = 15)
	# ax[3][2].set_yticklabels(ax[3][2].yaxis.get_majorticklabels(), rotation=0, size = 15)

	gs.tight_layout(fig)
	# plt.tight_layout()
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('visualizations', 'all_plots_{}.png'.format(name))
		fig.savefig(to_save, dpi=300)

def create_attendance_histogram(name, train, save = False):
	'''Creates and/or saves histogram of attendance for a given train as a subplot.

	'''
	sns.set_style("whitegrid")
	sns.set_palette(sns.color_palette('bright',12))

	bins = np.arange(4000, 26001, 500)
	fig, ax = plt.subplots(figsize=(20, 8))
	sns.distplot(train['Attendance'], ax=ax, kde=False, norm_hist=True, bins = bins)
	ax.set_xlabel('Attendance', fontsize = 20)
	ax.set_ylabel('Percent per Attendance', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.set_title(label = "NBA Attendance for {}".format(name), fontsize = 25)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'overall_attendance_hist_{}.png'.format(name))
		fig.savefig(to_save)

def create_daily_histogram(name, train, save = False):
	'''Creates and/or saves overlaid histograms of daily attendance as a subplot.

	'''

	sns.set_style("whitegrid")
	sns.set_palette(sns.color_palette('bright',12))

	bins = np.arange(4000, 26001, 500)
	fig, ax = plt.subplots(figsize=(20, 8))
	days = train['Day of Week'].unique()
	for day in days:
		sns.distplot(train.loc[train['Day of Week'] == day]['Attendance'],
							ax = ax, kde= False, norm_hist= True, bins = bins)
	ax.set_title(label = "NBA Attendance per Day for {}".format(name), fontsize = 25)
	ax.set_xlabel('Attendance', fontsize = 20)
	ax.set_ylabel('Percent per Attendance', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.legend(days,loc="upper right", fontsize=20)
	plt.show()

	if save:
		fig.savefig('daily_attendance_hist_{}.png'.format(name))

def create_daily_barchart(name, train, save = False): 
	'''Creates and/or saves horizontal bar chart of mean attendance per day as a subplot.

	'''

	fig, ax = plt.subplots(figsize=(20, 8))
	grouped = train[['Day of Week', 'Attendance']].groupby('Day of Week').mean()
	order = ['Monday', "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
	sns.barplot(x='Attendance', y=grouped.index, data=grouped, order=order, ci=None, orient='h', 
	             saturation=1, ax=ax, palette = sns.color_palette("cubehelix", 7))
	ax.set_xlim(16500,19000)
	ax.set_xticks(range(16500,19001,500))
	ax.set_xlabel('Average Attendance', fontsize = 20)
	ax.set_ylabel('Day of the Week', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.set_title(label = "Average NBA Attendance per Day for {}".format(name), fontsize = 25)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'daily_attendance_barchart_{}.png'.format(name))
		fig.savefig(to_save)

def create_monthly_histogram(name, train, save = False):
	'''Creates and/or saves overlaid histograms of monthly attendance as a subplot

	'''
	sns.set_style("whitegrid")
	sns.set_palette(sns.color_palette('bright',12))
	
	bins = np.arange(4000, 26001, 500)
	fig, ax = plt.subplots(figsize=(20, 8))
	months = train['Month'].unique()
	for month in months:
		sns.distplot(train.loc[train['Month'] == month]['Attendance'],
							ax = ax, kde= False, norm_hist= True, bins = bins)
	ax.set_title(label = "NBA Attendance per Month for {}".format(name), fontsize = 25)
	ax.set_xlabel('Attendance', fontsize = 20)
	ax.set_ylabel('Percent per Attendance', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.legend(months,loc="upper right", fontsize=20)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'monthly_attendance_hist_{}.png'.format(name))
		fig.savefig(to_save)

def create_monthly_barchart(name, train, save = False): 
	'''Creates and/or saves horizontal bar chart of mean attendance per day as a subplot.

	'''

	fig, ax = plt.subplots(figsize=(20, 8))
	grouped = train[['Month', 'Attendance']].groupby('Month').mean()
	order = ['October', 'November', 'December', 'January','February', 'March', 'April', 'May', 'June']
	sns.barplot(x='Attendance', y=grouped.index, data=grouped, order=order, ci=None, orient='h', 
	             saturation=1, ax=ax, palette = sns.color_palette("cubehelix", 9))
	ax.set_xlim(16500,19500)
	ax.set_xticks(range(16500,19501,500))
	ax.set_xlabel('Average Attendance', fontsize = 20)
	ax.set_ylabel('Month', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.set_title(label = "Average NBA Attendance per Month for {}".format(name), fontsize = 25)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'monthly_attendance_barchart_{}.png'.format(name))
		fig.savefig(to_save)


def create_yearly_histogram(name, train, save = False, years = np.arange(2014,2020)):
	'''Creates and/or saves overlaid histograms of yearly attendance of the last five years as a subplot.

	'''
	sns.set_style("whitegrid")
	sns.set_palette(sns.color_palette('bright',12))
	
	bins = np.arange(4000, 26001, 500)
	fig, ax = plt.subplots(figsize=(20, 8))
	for year in years:
		sns.distplot(train.loc[train.index.year == year]['Attendance'],
							ax = ax, kde= False, norm_hist= True, bins = bins)
	ax.set_title(label = "NBA Attendance per Year for {}".format(name), fontsize = 25)
	ax.set_xlabel('Attendance', fontsize = 20)
	ax.set_ylabel('Percent per Attendance', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.legend(years,loc="upper right", fontsize=20)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'yearly_attendance_hist_{}.png'.format(name))
		fig.savefig(to_save)

def create_yearly_barchart(name, train, save = False): 
	'''Creates and/or saves horizontal bar chart of mean attendance per day as a subplot.

	'''

	fig, ax = plt.subplots(figsize=(20, 8))
	train['Year'] = train.index.year
	grouped = train[['Year','Attendance']].groupby('Year').mean()
	order = np.arange(2004,2017)
	sns.barplot(x='Attendance', y=grouped.index, data=grouped, order=order, ci=None, orient='h', 
	             saturation=1, ax=ax, palette = sns.color_palette("cubehelix", 13))
	ax.set_xlim(16500,18500)
	ax.set_xticks(range(16500,18501,500))
	ax.set_xlabel('Average Attendance', fontsize = 20)
	ax.set_ylabel('Year', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.set_title(label = "Average NBA Attendance per Year for {}".format(name), fontsize = 25)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'yearly_attendance_barchart_{}.png'.format(name))
		fig.savefig(to_save)

def create_playoffs_histograms(name, train, save = False):
	'''Creates and/or saves overlaid histograms of attendance for playoff versus regular games as a subplot.

	'''
	sns.set_style("whitegrid")
	sns.set_palette(sns.color_palette('bright',12))
	
	bins = np.arange(4000, 26001, 500)
	fig, ax = plt.subplots(figsize=(20, 8))
	for j in [0,1]:
		sns.distplot(train.loc[train['Playoffs?'] == j]['Attendance'],
							ax = ax, kde= False, norm_hist= True, bins = bins)
	ax.set_title(label = "NBA Attendance for Regular and Playoff Games for {}".format(name), fontsize = 25)
	ax.set_xlabel('Attendance', fontsize = 20)
	ax.set_ylabel('Percent per Attendance', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.legend([0,1],loc="upper right", fontsize=20)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'playoffs_attendance_hist_{}.png'.format(name))
		fig.savefig(to_save)

def create_win_percent_histograms(name, train, save = False):
	'''Creates and/or saves overlaid histograms of attendance for different winning percentages 
	as a subplot.

	'''
	sns.set_style("whitegrid")
	sns.set_palette(sns.color_palette('bright',12))

	win_percent = np.arange(0,1,0.10)
	bins = np.arange(4000, 26001, 500)
	fig, ax = plt.subplots(figsize=(20, 8))
	for i in win_percent:
		sns.distplot(train.loc[(train['Curr Win %'] >= i) & (train['Curr Win %'] < i+0.1)]['Attendance'],
                        ax=ax, kde=False, norm_hist=True, bins = bins)
	ax.set_title(label = "NBA Attendance for Different Win Percentages for {}".format(name), fontsize = 25)
	ax.set_xlabel('Attendance', fontsize = 20)
	ax.set_ylabel('Percent per Attendance', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.legend(np.round(win_percent,1),loc="upper right", fontsize=20)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'win_percentage_attendance_hist_{}.png'.format(name))
		fig.savefig(to_save)

def create_last_five_record_histograms(name, train, save = False):
	'''Creates and/or saves overlaid histograms of attendance for different winning percentages 
	as a subplot.

	'''
	sns.set_style("whitegrid")
	sns.set_palette(sns.color_palette('bright',12))
	
	last_five = np.arange(0,6)
	bins = np.arange(4000, 26001, 500)
	fig, ax = plt.subplots(figsize=(20, 8))
	for i in last_five:
		sns.distplot(train.loc[train['Last Five'] == i]['Attendance'],
                        ax=ax, kde=False, norm_hist=True, bins = bins)
	ax.set_title(label = "NBA Attendance by Last Five Record for {}".format(name), fontsize = 25)
	ax.set_xlabel('Attendance', fontsize = 20)
	ax.set_ylabel('Percent per Attendance', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.legend(last_five,loc="upper right", fontsize=20)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'last_five_record_attendance_hist.{}.png'.format(name))
		fig.savefig(to_save)

def create_last_five_record_barchart(name, train, save = False): 
	'''Creates and/or saves horizontal bar chart of mean attendance per day as a subplot.

	'''

	fig, ax = plt.subplots(figsize=(20, 8))
	train['Year'] = train.index.year
	grouped = train[['Last Five','Attendance']].groupby('Last Five').mean()
	order = np.arange(0,6)
	sns.barplot(x='Attendance', y=grouped.index, data=grouped, order=order, ci=None, orient='h', 
	             saturation=1, ax=ax, palette = sns.color_palette("cubehelix", 5))
	ax.set_xlim(16500,18500)
	ax.set_xticks(range(16500,18501,500))
	ax.set_xlabel('Average Attendance', fontsize = 20)
	ax.set_ylabel('Record Over Last Five Games', fontsize = 20) 
	ax.tick_params(labelsize=15)
	ax.set_title(label = "Average NBA Attendance per Last Five Record for {}".format(name), fontsize = 25)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'last_five_attendance_barchart_{}.png'.format(name))
		fig.savefig(to_save)

def create_heatmap(name, train, save = False):
	'''Creates and/or saves a heatmap of the correlation between the numerical variables of a train.

	'''
	train = train.copy()
	num_numerical = ds.get_number_numerical(name)
	train_num = train.iloc[:,0:num_numerical]
	fig, ax = plt.subplots(figsize=(15, 15))
	heat = sns.heatmap(train_num.corr(),annot = True, ax = ax, fmt = '.2f', 
	                   cbar = True, square = True, xticklabels= True, yticklabels = True,
	                  annot_kws={'size':16}, cmap = 'coolwarm', center= 0, vmin=-1, vmax=1,
	                  cbar_kws={"shrink": .82})
	ax.set_title('Heatmap of Numerical Variable Correlation for {}'.format(name), size=25) 
	plt.yticks(rotation=0,size = 15) 
	plt.xticks(rotation=30, size = 15)
	ax.collections[0].colorbar.ax.tick_params(labelsize=15)

	# Make annotations larger if abs(correlation) above 0.2
	num_corrs = len(np.unique(train_num.corr().values.flatten()))
	bigs = []
	for i in np.arange(2,num_corrs+1):
	    val = round(np.sort(np.abs(np.unique(train_num.corr().values.flatten())))[-i],2)
	    if val > 0.2:
	        bigs = np.append(bigs, val)
	for text in heat.texts:
	    num =  pd.to_numeric(text.get_text())
	    i = np.where(bigs == abs(num))[0]
	    if i.size > 0:
	        text.set_color('white')
	        text.set_size(40-(i[0]*3))
	plt.show()  

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', '{}_heatmap.png'.format(name))
		fig.savefig(to_save) 






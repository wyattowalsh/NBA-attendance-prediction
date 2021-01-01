import numpy as np
import pandas as pd
import src.data.datasets as ds
import src.data.train_test_split as split
import src.models.metrics as metrics
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def pca(name, X_train, X_test, dimension):
	"""

	"""

	X_train = X_train.copy()
	X_test = X_test.copy()
	num_numerical = ds.get_number_numerical(name)
	X_train_s = split.standardize(name, X_train)
	X_test_s = split.standardize(name, X_test)
	X_train_s_numerical = X_train_s.iloc[:,0:num_numerical]
	X_train_s_categorical = X_train_s.iloc[:,num_numerical:]
	X_test_s_numerical = X_test_s.iloc[:,0:num_numerical]
	X_test_s_categorical = X_test_s.iloc[:,num_numerical:]
	estimator = PCA(dimension)
	X_train_s_numerical_reduced = pd.DataFrame(estimator.fit_transform(X_train_s_numerical), 
	                                           index = X_train_s_categorical.index)
	X_test_s_numerical_reduced = pd.DataFrame(estimator.transform(X_test_s_numerical), 
	                                          index = X_test_s_categorical.index)
	X_train_s = pd.concat([X_train_s_numerical_reduced, X_train_s_categorical], axis = 1)
	X_test_s = pd.concat([X_test_s_numerical_reduced, X_test_s_categorical], axis = 1)
	return X_train_s, X_test_s


def pca_cv(name, save = False):
	'''

	'''

	display_name = ds.get_names()[name]
	X_train, X_test, y_train, y_test, train = split.split_subset(name)
	num_numerical = ds.get_number_numerical()[name]
	X_train_s, X_test_s = split.standardize(name, X_train, X_test)
	X_train_s_numerical = X_train_s.iloc[:,0:num_numerical]
	X_train_s_categorical = X_train_s.iloc[:,num_numerical:]
	X_test_s_numerical = X_test_s.iloc[:,0:num_numerical]
	X_test_s_categorical = X_test_s.iloc[:,num_numerical:]
	df = pd.DataFrame()
	ols = LinearRegression()
	ev = []
	for i in np.arange(1,num_numerical):
		pca = PCA(i, random_state = 18)
		X_train_s_numerical_reduced = pd.DataFrame(pca.fit_transform(X_train_s_numerical), 
	                                     	  index = X_train_s_categorical.index)
		X_test_s_numerical_reduced = pd.DataFrame(pca.transform(X_test_s_numerical), 
	                                          index = X_test_s_categorical.index)
		X_train_s = pd.concat([X_train_s_numerical_reduced, X_train_s_categorical], axis = 1)
		X_test_s = pd.concat([X_test_s_numerical_reduced, X_test_s_categorical], axis = 1)

		model = ols.fit(X_train_s, y_train)
		preds = model.predict(X_test_s)
		preds = metrics.apply_metrics('{}: {} dimensions'.format(display_name, i), y_test, preds.ravel(),y_train)
		df = pd.concat([df, preds], axis = 0)
		ev.append(1-sum(pca.explained_variance_))

	if save:
		to_save = Path().resolve().joinpath('features', 'pca', '{}.csv'.format(name))
		df.to_csv(to_save)

	return df, ev

def pca_scree(names, save = False):
	"""

	"""


	name_dict = ds.get_names()
	fig, ax = plt.subplots(nrows = 1, ncols= len(names), figsize=(40, 20))
	for i, name in enumerate(names): 
		y_values = decomp.pca_cv(name)[1]
		print(y_values)
		sns.lineplot(x = np.arange(1, len(y_values)+1), y = y_values, markers = 'o', ax = ax[i])
		ax[i].tick_params(labelsize=15)
		ax[i].set_xlabel('Number of Components', fontsize=20)
		ax[i].set_ylabel('Cumulative Explained Variance', fontsize=20);
		ax[i].set_title('{}'.format(name_dict[name]))
	plt.tight_layout()
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', 'pca_{}.png'.format(names[1]))
		fig.savefig(to_save)  


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, make_scorer, mean_squared_error # explained_variance_score

def create_metrics():
	'''Creates metrics that functions and the user can use.

	'''

	# def mean_absolute_percentage_error(y_true, y_pred): 
		# return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
	def root_mean_square_error(y_true, y_pred):
		return np.sqrt(mean_squared_error(y_true, y_pred))
	def mean_absolute_error_custom(y_true, y_pred):
		return np.abs(mean_absolute_error(y_true,y_pred))
	def out_of_sample_r2(y_true_test,y_pred, y_true_train):
		sse_t = sum((y_true_test-y_pred)**2)
		sst_t = sum((y_true_test-y_true_train.mean())**2)
		return 1-(sse_t/sst_t)


	osr2 = make_scorer(out_of_sample_r2, greater_is_better = True)
	# evs = make_scorer(explained_variance_score)
	mae_custom = make_scorer(mean_absolute_error_custom, greater_is_better = False)
	rmse_custom = make_scorer(root_mean_square_error, greater_is_better = False)
	# mape_custom = make_scorer(mean_absolute_percentage_error, greater_is_better = False)
	to_score = {'$OSR^2$': osr2,
			   "Mean Absolute Error": mae_custom, "Root Mean Square Error": rmse_custom,
			   }
	scoring = {'$OSR^2$': out_of_sample_r2,
			   "Mean Absolute Error": mean_absolute_error_custom, 
			   "Root Mean Square Error": root_mean_square_error,
			  }
	# "Explained Variance Score": evs "Mean Absolute Percent Error": mape_custom
	#"Explained Variance Score": explained_variance_score "Mean Absolute Percent Error": mean_absolute_percentage_error

	return to_score, scoring


def apply_metrics(name, y_true, y_pred, y_true_train):
	"""

	"""
	
	y_true = y_true.copy()
	y_pred = y_pred.copy()
	y_true_train = y_true_train.copy()
	scoring = create_metrics()[1]
	keys = list(scoring.keys())
	scores = {}
	for metric in keys:
		if metric == '$OSR^2$':
			score = scoring[metric](y_true, y_pred, y_true_train)
		else:
			score = scoring[metric](y_true, y_pred)	
		scores[metric] = score
	scores = pd.DataFrame(scores.values(), index = keys, columns = [name]).transpose()
	return scores



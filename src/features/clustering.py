import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import src.data.datasets as ds
import src.data.train_test_split as split
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def kmeans(name, n_clusters, X_train, X_test):
	'''

	'''

	X_train = X_train.copy()
	X_test = X_test.copy()
	num_numerical = ds.get_number_numerical()[name]
	X_train_s_numerical = split.standardize(name, X_train, X_test)[0].iloc[:,0:num_numerical]
	return KMeans(n_clusters= n_clusters, random_state=18, n_jobs = -1).fit(X_train_s_numerical)


def elbow_method_kmeans(name, max_clusters = 30, min_clusters = 2, save = False):
	'''Creates elbow method plot for varying number of clusters.

	Y-axis is the sum of squared distances of samples to their closest cluster center.
	'''

	display_name = ds.get_names()[name]
	X_train, X_test= split.split_subset(name)[0:1]
	num_numerical = ds.get_number_numerical()[name]
	X_train_numerical = X_train.iloc[:,0:num_numerical]
	distortions = []
	cluster_range = range(min_clusters,max_clusters)
	for clusters in cluster_range:
		kmean = kmeans(name, clusters, X_train_numerical)
		distortions.append(kmean.inertia_)

	fig, ax = plt.subplots(figsize=(10, 10))
	ax.plot(cluster_range, distortions, marker = 'o')
	ax.set_title('Elbow Method for KMeans for {}'.format(display_name), size=25)
	ax.set_xlabel('Number of Clusters', fontsize = 20)
	ax.set_ylabel('Sum of Squared Distances', fontsize = 20)
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('visualizations', 'clustering_elbow_{}.png'.format(name))
		fig.savefig(to_save, dpi=300) 

def silhouettes(name, X_train, max_clusters = 15, min_clusters = 5, save = False):
	'''

	'''

	X_train = X_train.copy()
	num_numerical = ds.get_number_numerical(name)
	X_train_s_numerical = split.standardize(name, X_train).iloc[:,0:num_numerical]
	cluster_range = range(min_clusters,max_clusters+1)
	for clusters in cluster_range:
		fig, ax = plt.subplots()

		fig.set_size_inches(18, 7)

		# The 1st subplot is the silhouette plot
		# The silhouette coefficient can range from -1, 1 but in this example all
		# lie within [-0.1, 1]
		ax.set_xlim([-0.7, 1])
		# The (n_clusters+1)*10 is for inserting blank space between silhouette
		# plots of individual clusters, to demarcate them clearly.
		ax.set_ylim([0, len(X_train_s_numerical) + (clusters + 1) * 10])

		cluster_labels = kmeans(name, clusters, X_train_s_numerical).predict(X_train_s_numerical)
		silhouette_avg = silhouette_score(X_train_s_numerical, cluster_labels)
		print("For n_clusters =", clusters, "The average silhouette_score is :", silhouette_avg)

		cluster_silhouette = silhouette_samples(X_train_s_numerical, cluster_labels)

		y_lower = 10
		for i in range(clusters):
			ith_cluster_silhouette_values = cluster_silhouette[cluster_labels == i]
			ith_cluster_silhouette_values.sort()

			size_cluster_i = ith_cluster_silhouette_values.shape[0]
			y_upper = y_lower + size_cluster_i

			color = cm.nipy_spectral(float(i) / clusters)
			ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, 
								facecolor=color, edgecolor=color, alpha=0.7)

			# Label the silhouette plots with their cluster numbers at the middle
			ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

			# Compute the new y_lower for next plot
			y_lower = y_upper + 10  # 10 for the 0 samples

		ax.set_title("The silhouette plot for the various clusters.")
		ax.set_xlabel("The silhouette coefficient values")
		ax.set_ylabel("Cluster label")

		# The vertical line for average silhouette score of all the values
		ax.axvline(x=silhouette_avg, color="red", linestyle="--")

		ax.set_yticks([])  # Clear the yaxis labels / ticks
		ax.set_xticks(np.arange(-0.6,1.1,0.2))
		plt.show()

	if save:
		to_save = Path().resolve().joinpath('data', 'visualizations', '{}_elbow.png'.format(name))
		fig.savefig(to_save) 

def create_cluster_plots(name, save = False): 
	'''

	'''

	X_train,X_test = split.split_subset(name)[0:2]
	num_numerical = ds.get_number_numerical()[name]
	X_train_numerical = X_train.iloc[:,0:num_numerical]
	X_test_numerical = X_test.iloc[:,0:num_numerical]
	distortions = []
	cluster_range = range(2,31)
	for clusters in cluster_range:
		kmean = kmeans(name, clusters, X_train, X_test)
		distortions.append(kmean.inertia_)

	fig, ax = plt.subplots(ncols = 2, figsize=(40, 12))
	ax[0].plot(cluster_range, distortions, marker = 'o')
	ax[0].set_title('KMeans Scree Plot', fontsize = 40)
	ax[0].set_xlabel('Number of Clusters', fontsize = 30)
	ax[0].set_ylabel('Sum of Squared Distances', fontsize = 30)
	ax[0].tick_params(labelsize=20)
	for i, txt in enumerate(cluster_range):
		annot = ax[0].annotate('{}'.format(txt), (cluster_range[i],distortions[i]))
		annot.set_fontsize(25)

	X_train_s_numerical = split.standardize(name, X_train_numerical, X_test_numerical)[0]
	clusters = 7
	if name == 'dataset_3':
		clusters = 9
	ax[1].set_xlim([-0.3, 0.8])
	ax[1].set_ylim([0, len(X_train_s_numerical) + (clusters + 1) * 10])

	cluster_labels = kmeans(name, clusters, X_train, X_test).predict(X_train_s_numerical)
	silhouette_avg = silhouette_score(X_train_s_numerical, cluster_labels)

	cluster_silhouette = silhouette_samples(X_train_s_numerical, cluster_labels)

	y_lower = 10
	for i in range(clusters):
		ith_cluster_silhouette_values = cluster_silhouette[cluster_labels == i]
		ith_cluster_silhouette_values.sort()

		size_cluster_i = ith_cluster_silhouette_values.shape[0]
		y_upper = y_lower + size_cluster_i

		color = cm.nipy_spectral(float(i) / clusters)
		ax[1].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, 
							facecolor=color, edgecolor=color, alpha=0.7)

		ax[1].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize = 25)

		y_lower = y_upper + 10  

	ax[1].set_title("Silhouette plot for {} clusters.".format(clusters) , fontsize = 40)
	ax[1].set_xlabel("Silhouette Coefficient Values",fontsize = 30)
	ax[1].set_ylabel("Cluster label",fontsize = 30)

	ax[1].axvline(x=silhouette_avg, color="red", linestyle="--")

	ax[1].set_yticks([]) 
	ax[1].set_xticks(np.arange(-0.3,0.9,0.1))
	ax[1].tick_params(labelsize=20)

	plt.tight_layout()
	plt.show()

	if save:
		to_save = Path().resolve().joinpath('visualizations', 'clustering', '{}.png'.format(name))
		fig.savefig(to_save, dpi = 300) 
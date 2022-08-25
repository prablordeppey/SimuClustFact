import numpy as np

def Screeplot(errors, tol=10e-3):
	"""
	proposes the optimal index for a monotonic decreasing error list

	Args:
		errors (list, ndarray): list of errors in ascending order
		tol (float, optional): acceptable tolerance level. smaller tol, larger proposals. Defaults to 1e-1 (10%)

	Returns:
		int: optimal number of clusters or factor of interest
	"""
	# initialization
	max_error = np.max(errors)  # maximum error in wcss_
	index = 1
	for (g,error) in enumerate(errors):
		percent_error = error/max_error
		index = index*percent_error
		if index <= tol:
			break
	G = g+1  # python indexing starts from 0
	return G


def RuleOfThumb(n):
	"""
	Rule of thumb index

	Args:
		n (int): number of observations

	Returns:
		int: optimal number of clusters
	"""
	k = int(np.power(n/2, 0.5))
	return k


def PseudoF(bss, wss, tensor_shape, rank):
	"""
	The pseudo F statistic describes the ratio of between-cluster variance to within cluster variance.
	Large values of Pseudo F indicate close-knit and separated clusters.
	In particular, peaks in the pseudo F statistic are indicators of greater cluster separation.
	This implementation is a generalization provided by Rocci and Vichi.

	Args:
	-----
		bss (float): between cluster sum of squares deviance
		wss (float): within cluster sum of squares deviance
		tensor_shape (tuple): (I,J,K) tensor dimensions
		rank (tuple): (G,Q,R) G clusters, Q components for variables, R components for Occasions
	
	Reference:
	----------
		...	[1] T. Caliński & J Harabasz (1974).
			A dendrite method for cluster analysis
			Communications in Statistics, 3:1, 1-27, DOI: 10.1080/03610927408827101

		..	[2] Roberto Rocci and Maurizio Vichi (2005).
			Three-mode component analysis with crisp or fuzzy partition of units. 
			Psychometrika, 70:715–736, 02 2005.
	"""
	I,J,K = tensor_shape
	G,Q,R = rank
	db = (G-1)*Q*R + (J-Q)*Q + (K-R)*R
	dw = I*J*K - G*Q*R + (J-Q)*Q + (K-R)*R
	return (bss/db)/(wss/dw)


def RMSSTD(wss,n,v):
	"""
	Root Mean Square Standard Deviation (RMSSTD) Index.
	This index is the root mean square standard deviation of all the variables within each cluster.
	calculating the within-group sum of squares of each cluster and normalizing it by the product of the number of elements in the cluster and the number of variables 

	Args:
	-----
		wss (float): within sum of squares deviance
		n (int): number of objects in cluster.
		v (int): number of variables.

	References:
	-----------
		...	Rujasiri, Piyatida & Chomtee, Boonorm. (2009).
			Comparison of Clustering Techniques for Cluster Analysis.
			Kasetsart Journal - Natural Science. 43. 
	"""
	return np.sqrt(wss/(v*(n-1)))
	

def Dunn(X, centroids, labels):
	"""
	Dunn’s index is defined as the minimum of the ratio of the dissimilarity measure between two clusters
	to the diameter of cluster, where the minimum is taken over all the clusters in the data set.
	The clustering which attains the maximum in the plot of Dunn’s versus the number of clusters, is the appropriate one.

	Args:
	-----
		centroids (ndarray): matrix of cluster centroid.
	"""
	
	# obtain cluster index, with cluster points
	unique_labels = np.unique(labels)
	centroid_points = {_:[] for _ in unique_labels}
	for i in range(X.shape[0]):
		centroid_points[labels[i]].append(i)

	# compute dunn values
	dunn = []
	for c_index in range(centroids.shape[0]):
		point_dists = X[centroid_points[c_index]] - centroids[c_index,]
		c_diameter = 2*np.linalg.norm(point_dists, 2, axis=1).max()  # cluster diameter
		centroid_dists = centroids - centroids[c_index]
		dunn.append(sorted(np.linalg.norm(centroid_dists,2,axis=1)/c_diameter)[1])
	
	return np.min(dunn)


	

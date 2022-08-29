### -------------------

# Helping functions to 

### -------------------

import numpy as np
from tensor import fold, unfold


# =========== 1 RUN OF THE KMEANS ALGORITHM

def OneKMeans(Y_i_qr, G, U_i_g=None, rng=None, seed=None):
	"""
	Performs 1 run of the KMeans algorithm in the reduced space to update U.
	To deal with empty clusters:
	1. split the biggest cluster into two (time consuming)
	2. repeat the clustering process until all clusters have points.

	Args:
		Y_i_qr (ndarray): IxQR component scores matrix.
		G (int): number of clusters to group objects.
		rng (rng): random number generator.
		seed (int): used inplace of rng 

	Returns:
		ndarray: IxG membership function matrix, U_i_g.
	"""

	# defining random generator with no seed to radnom results.
	if rng is None:
		rng = np.random.default_rng(seed)

	Y_i_qr = np.array(Y_i_qr)

	I = Y_i_qr.shape[0]

	# initialize centroids matrix
	if U_i_g is None:
		U_i_g = RandomMembershipMatrix(I,G,rng=rng)
	else:
		U_i_g = np.array(U_i_g)
	Y_g_qr = np.linalg.inv(U_i_g.T@U_i_g) @ U_i_g.T @ Y_i_qr

	# ------- case 1: repeat until no empty clustering	
	U_i_g = np.zeros((I,G))

	def split_update(LC, EC, U_i_g):
		"""
		Splits a given cluster of points into 2 subclusters.
		The membership matrix U_i_g is updated and returned inturn.

		Args:
			LC (int): cluster with the largest number of objects.
			EC (int): empty cluster to share LC members with.
			U_i_g (int): stochastic membership matrix to update.

		Returns:
			ndarray: binary stochastic membership matrix.
		"""
		M = int(C_g[LC])  # number of objects in the chosen largest cluster.
		cluster_members_indices = np.where(U_i_g[:,LC])[0]  # indices of objects in the largest cluster

		U_m_2 = RandomMembershipMatrix(I=M, G=2, rng=rng)  # initialize matrix with 2 groups
		Y_2_qr = np.linalg.inv(U_m_2.T@U_m_2) @ U_m_2.T @ Y_i_qr[cluster_members_indices,:]  # 2xQR centroids matrix for subclusters

		# assign each cluster member to the respective sub-cluster
		for i in cluster_members_indices:
			dist = ((Y_i_qr[i,]-Y_2_qr)**2).sum(axis=1)  # calculate distance between obj and the 2 sub-centroids.
			min_dist_cluster = dist.argmin()  # get cluster with smallest distance.

			# obj reassignment
			if min_dist_cluster == 0:
				U_i_g[i, LC] = 0  # unassign the obj from that cluster.
				U_i_g[i, EC] = 1  # assign the obj to that cluster.
			
		return U_i_g

	# assign each object to the respective cluster
	for i in range(I):
		dist = ((Y_i_qr[i,:]-Y_g_qr)**2).sum(axis=1)  # calculate distance between obj and centroids.
		min_dist_cluster = dist.argmin()  # get cluster with smallest distancee from object.
		U_i_g[i, min_dist_cluster] = 1  # assign the object to that cluster.

	# possibility of observing empty clusters
	C_g = U_i_g.sum(axis=0)  # get count of members in each cluster
	while (C_g==0).any():
		LC = C_g.argmax()  # select the largest cluster
		EC = np.where(C_g==0)[0][0]  # select next empty cluster
		U_i_g = split_update(LC, EC, U_i_g)  # splitting cluster into 2 sub-clusters and updating U_i_g
		C_g = U_i_g.sum(axis=0)  # ensure the alg stops

	return U_i_g

# ===========  END OF KMEANS ALGORITHM


# =========== BUILDING MEMBERSHIP FUNCTION MATRIX CONSTRUCTION

def RandomMembershipMatrix(I, G, rng=None, seed=None):
	"""
	Generates U_i_g membership function matrix.
	U_i_g is a binary stochastic matrix with rows summing to 1.

	Args:
		I (int): number of objects/units.
		G (int): number of groups/clusters.
		rng (rng): random number generator.
		seed (int): used inplace of rng

	Returns:
		ndarray: (IxG) matrix with only one 1 in each row.
	"""
	# defining random generator with no seed to radnom results.
	if rng is None:
		rng = np.random.default_rng(seed)

	U_i_g = np.zeros((I,G))  # initialize U_i_g to 0's.
	U_i_g[:G,] = np.eye(G)  # first G assignments to unique clusters. To ensure no cluster is empty

	# assign random clusters to remaining objects
	if I > G:
		for p in range(G,I):
			c = rng.integers(G) # choose a random cluster for the i'th object.
			U_i_g[p,c] = 1  # assign object i to cluster p

	return U_i_g

# ===========  END OF MEMBERSHIP FUNCTION CONSTRUCTION


# ===========  LARGEST EIGENVECTORS

def SingularVectors(X, D):
	"""
	Returns first D left singular vectors of a given matrix
	either via svd as currently implemented or eig function (commented out)
	both approaches have no effect on the obtained results.

	Args:
	-----
	X (ndarray): matrix
	D (int): number of singular vectors to extract.
	"""
	u,_,_ = np.linalg.svd(X)
	return u[:,:D]

	# eigenValues,eigVectors = np.linalg.eig(X)
	# idx = eigenValues.argsort()[::-1]   
	# eigenValues = eigenValues[idx]
	# return eigVectors[:,:D]

# ===========  END OF LARGEST EIGENVECTORS


# ===========  START OF THE PseudoF STATISTIC FUNCTION

def PseudoF(bss, wss, full_tensor_shape, reduced_tensor_shape):
	"""
	The pseudo F statistic describes the ratio of between-cluster variance to within cluster variance.
	Large values of Pseudo F indicate close-knit and separated clusters.
	In particular, peaks in the pseudo F statistic are indicators of greater cluster separation.
	This implementation is a generalization provided by Rocci and Vichi.

	Args:
	-----
		bss (float): between cluster sum of squares deviance
		wss (float): within cluster sum of squares deviance
		full_tensor_shape (tuple): (I,J,K) tensor dimensions
		reduced_tensor_shape (tuple): (G,Q,R) G clusters, Q components for variables, R components for Occasions
	
	Reference:
	----------
		...	[1] T. Caliński & J Harabasz (1974).
			A dendrite method for cluster analysis
			Communications in Statistics, 3:1, 1-27, DOI: 10.1080/03610927408827101

		..	[2] Roberto Rocci and Maurizio Vichi (2005).
			Three-mode component analysis with crisp or fuzzy partition of units. 
			Psychometrika, 70:715–736, 02 2005.
	"""
	I,J,K = full_tensor_shape
	G,Q,R = reduced_tensor_shape
	db = (G-1)*Q*R + (J-Q)*Q + (K-R)*R
	dw = I*J*K - G*Q*R + (J-Q)*Q + (K-R)*R
	return (bss/db)/(wss/dw)

# ===========  END OF PseudoF STATISTIC FUNCTION


# ===========  START OF _BASECLASS

class _BaseClass:
	"""
	Base class for the tandem tucker-factorial and kmeans-clustering models.
	For checking initialization configuration.
	
	Initialization Args:
	--------------------
	n_max_iter (int, optional): maximum number of iterations. Defaults to 10.
	n_loops (int, optional): number of random initializations to gurantee global results. Defaults to 10.
	tol (float, optional): tolerance level/acceptable error. Defaults to 1e-5.
	random_state (int, optional): seed for random sequence generation. Defaults to None.
	verbose (bool, optional): whether to display executions output or not. Defaults to False.'''
	"""

	def __init__(
		self,
		random_state=None,
		verbose=False,
		init='svd',
		n_max_iter=10,
		n_loops=10,
		tol=1e-2,
		U_i_g=None,
		B_j_q=None,
		C_k_r=None 
	):
		self.init = init
		self.n_max_iter = n_max_iter
		self.n_loops = n_loops
		self.tol = tol
		self.random_state = random_state
		self.verbose = verbose

		self.full_tensor_shape = None  # (I,J,K)
		self.reduced_tensor_shape = None  # (G,Q,R)
		self.B_j_q = B_j_q
		self.C_k_r = C_k_r
		self.U_i_g = U_i_g

	# Check initialization configurations.
	def _check_params(self):
		"""
		verify valid input tensor dimensions in the full and reduced space.
		"""
		# n_max_iter
		if (not isinstance(self.n_max_iter, int)) or (not self.n_max_iter > 0):
			raise ValueError(f"n_max_iter should be > 0, got {self.n_max_iter} instead.")

		# tol
		if (not isinstance(self.tol, float)) or (not 0 < self.tol < 1):
			raise ValueError(
				f"tolerance should be very small positive number between < 1 but got {self.tol}"
			)

		# verbose
		if not isinstance(self.verbose, bool):
			raise ValueError(
				f"verbose must be boolean but got {type(self.verbose).__name__}"
			)

		# tensor dimension in full and reduced space must be 1D array.
		if np.array(self.full_tensor_shape).dtype != 'int32' or np.array(self.reduced_tensor_shape).dtype != 'int32':
			raise ValueError(
				f"please ensure all dimensions are positive integers."
			)

		# ensure shape variables are 1D array with 3 integers
		if np.array(self.full_tensor_shape).size != 3 or np.array(self.reduced_tensor_shape).size != 3:
			raise ValueError(
				f"full and reduced tensor dimensions must have only 3 elements, but got \
					|full_shape|={len(self.full_tensor_shape)} and |reduced_shape|={len(self.reduced_tensor_shape)}"
			)

		# ensure dimensions are positive integers
		if (not (np.array(self.full_tensor_shape) > 0).all()) or (not (np.array(self.reduced_tensor_shape) > 0).all()):
			raise ValueError(
				f"please ensure all dimensions are positive integers."
			)

		# ensure all dimensions given are greater than 0
		if not (np.array(self.reduced_tensor_shape) <= np.array(self.full_tensor_shape)).all():
			raise ValueError(
				f"reduced_tensor_shape={self.reduced_tensor_shape} must be <= full_tensor_shape={self.full_tensor_shape}."
			)

	# component matrices validation
	def _check_initialized_components(self):
		"""
		If U_i_g,B_j_q and C_k_r are user-defined, dimensions of these matrices are validated.
		U_i_g must also be row-stochastic (row-sums equal to 1)
		"""
		# check U membership matrix
		if self.U_i_g is not None:
			# check dimension
			if self.U_i_g.shape != (self.full_tensor_shape[0], self.reduced_tensor_shape[0]):
				raise ValueError(
					f"incorrect U_i_g matrix, expected shape {(self.full_tensor_shape[0], self.reduced_tensor_shape[0])} but got {self.U_i_g.shape}"
				)
			
			# check row stochastic nature
			if (self.U_i_g.sum(axis=1) != 1).all():
				raise ValueError(
					f"incorrect U_i_g matrix. U_i_g must be row stochastic."
				)

		# check B component matrix
		if self.B_j_q is not None:
			# check dimension
			if self.B_j_q.shape != (self.full_tensor_shape[1], self.reduced_tensor_shape[1]):
				raise ValueError(
					f"incorrect B_j_q matrix, expected shape {(self.full_tensor_shape[1], self.reduced_tensor_shape[1])} but got {self.B_j_q.shape}"
				)
		
		# check C component matrix
		if self.C_k_r is not None:
			# check dimension
			if self.C_k_r.shape != (self.full_tensor_shape[2], self.reduced_tensor_shape[2]):
				raise ValueError(
					f"incorrect C_k_r matrix, expected shape {(self.full_tensor_shape[2], self.reduced_tensor_shape[2])} but got {self.C_k_r.shape}"
				)

# ===========  END OF _BASECLASS
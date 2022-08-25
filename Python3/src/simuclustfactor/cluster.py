# import numpy as np
# from sklearn.cluster import k_means
# from utils import construct_membership_matrix

# ### === ACTUAL KMEANS IMPLEMENTATION
# class KMeans:

# 	"""
# 	An implementation of the KMeans algorithm. KMeans++ initialization of the centroids is considered.
# 	Random initialization could be specified via the init attribute.

# 	Init Args:
# 	----------
# 	init (str, optional): initialization method, ['k-means++', 'random']. Defaults to 'k-means++'
# 	n_inits (int, optional): number of runs to gurantee global results. Defaults to 10.
# 	random_state (int, optional): seed for random number generation. Defaults to None
# 	rng (random._generator.Generator): random number generator object. Defaults to None.
# 	verbose (bool, optional): Verbosity. Defaults to False

# 	Fit Args:
# 	---------
# 	X_i_j (ndarray): (I,J) matrix to be clustered. I objects and J variables.
# 	G (int): number of classes to consider.

# 	Returns:
# 	--------
# 	membership_matrix_ (ndarray): (I,G) binary stochastic membership matrix of I objects to G clusters (U_i_g).
# 	centroids_ (ndarray): (G,J) matrix of centroids for clusters.
# 	labels_ (list): (I,1) array of labels for the I objects
# 	wss_ (float): within cluster same of squared deviations.
# 	bss_ (float): between cluster sum of squared deviations.
# 	tss_ (float): total sum of squared deviations.
# 	explained_variance (float): explained variance.
# 	"""

# 	def __init__(
# 		self,
# 		init='k-means++',
# 		n_inits=10,
# 		tol=1e-5,
# 		random_state=None,
# 		rng=None,
# 		verbose=False,
# 	):
# 		self.tol = tol
# 		self.init = init
# 		self.n_inits = n_inits
# 		self.random_state = random_state
# 		self.rng = rng
# 		self.verbose = verbose
# 		self.X = None
# 		self.n_clusters = None

# 	# check data matrix X and number of clusters G
# 	def _check_params(self, X_shape, G):
# 		# check G
# 		if not isinstance(G, int):
# 			raise ValueError(
# 				f"expected G (number of clusters) as integer but got a {type(G)}"
# 			)
		
# 		if not G <= X_shape[0]:
# 			raise ValueError(
# 				f"G (number of clusters) must be <= {X_shape[0]}"
# 			)

# 		# check X
# 		if len(X_shape) != 2:
# 			raise ValueError(
# 				f"X must be a valid matrix of shape (I,J) but got shape {X_shape}"
# 			)

# 	# Fitting the KMeans model.
# 	def fit(self, **vars):
# 		"""
# 		Args:
# 		-----
# 			**vars (dict): dictionary of inputs to the KMeans method.
# 				- X (ndarray): (I,J) matrix to be clustered.
# 				- n_clusters (int): number of clusters to find
# 		"""
		
# 		# fetch X and G
# 		if vars['X'] is None: raise(ValueError('specify matrix X'))
# 		if vars['n_clusters'] is None: raise(ValueError('specify n_clusters'))

# 		X = vars['X']
# 		G = vars['n_clusters']

# 		X = (X - X.mean(axis=0, keepdims=True))/X.std(axis=0, keepdims=True)

# 		# defining random generator with seed
# 		if self.rng is None:
# 			self.rng = np.random.default_rng(self.random_state)

# 		self._check_params(X.shape, G)

# 		kmeans = k_means(
# 			init=self.init,
# 			n_init=self.n_inits,
# 			tol=self.tol,
# 			random_state=int(self.rng.random()*10),
# 			return_n_iter=True,
# 			**vars
# 			)

# 		centroids = kmeans[0]
# 		labels = kmeans[1]
# 		wss = kmeans[2]
# 		self.best_init_ = kmeans[-1]

# 		U_i_g = construct_membership_matrix(labels)

# 		Z = U_i_g @ centroids		
# 		total_variance = sum([X[:,variable].var() for variable in range(X.shape[1])])
# 		explained_variance = sum([Z[:,component].var() for component in range(Z.shape[1])])

# 		# sum of squares
# 		bss = (Z).var()*(Z).size
# 		if np.isclose(bss,0): bss = 0
# 		tss = wss+bss

# 		self.centroids_ = centroids
# 		self.membership_matrix_ = U_i_g
# 		self.labels_ = labels

# 		self.tss_ = tss
# 		self.wss_ = wss
# 		self.bss_ = bss
# 		self.explained_variance = explained_variance

# 		# output to std output/screen
# 		if self.verbose:
# 			bss_percent = (bss/tss)*100
# 			wss_percent = 100 - bss_percent
# 			explained_percent = (explained_variance/total_variance)*100
# 			print('total_variance:', total_variance)
# 			print('explained_variance:', f'{explained_variance} ({explained_percent})%')
# 			print('tss:', f'{tss}')
# 			print('bss:', f'{bss} ({bss_percent})%')
# 			print('wss:', f'{wss} ({wss_percent})%')

# 		return self

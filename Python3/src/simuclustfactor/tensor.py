import numpy as np

def unfold(tensor, mode):
	"""
	Args:
		tensor (ndarray): (K,I,J) three-way tensor
		mode (int): mode to unfold by. 0->k, 1->i, 2->j

	Returns:
		ndarray: matricized tensor along the given mode

	Example:
		# mode 0 unfolding
		>> fold(X_i_j_k, mode=0)
		>> X_i_jk

		# mode 1 unfolding
		>> fold(X_i_j_k, mode=1)
		>> X_i_jk

		# mode 2 unfolding
		>> fold(X_i_j_k, mode=2)
		>> X_j_ki
	"""
	tensor = np.array(tensor)

	# chceks
	if not isinstance(mode, int): raise ValueError(f'mode is expected to be a number but got {{type(mode).__name__}}')
	if mode not in range(len(tensor.shape)): raise ValueError(f'possible modes are {range(len(tensor.shape))}. refer to the manual for appropriate valid mode.')  # only valid specified mode
	
	# main
	if (mode==0):  # (K,IJ)
		unfolded = np.array([X.flatten() for X in tensor])
	if (mode==1):  # (I,JK)
		unfolded = np.concatenate(tensor, axis=1)
	if (mode==2):  # (J,KI)
		unfolded = []
		for i in range(tensor.shape[1]):
			face = []
			for k in range(tensor.shape[0]):
				face.append(tensor[k][i])
			face = np.array(face).T
			unfolded.append(face)
		unfolded = np.hstack(unfolded)

	return unfolded


def fold(X, mode, shape):
	"""
	performs correct folding operation from matrix to tensor.

	Args:
		X (ndarray): matrix to be folded into tensor
		mode (int): mode along which to fold matrix
		shape (tuple): (K,G,J) tensor shape of of input.

	Returns:
		ndarray: (K,I,J) folded matrix / tensor

	Example:
		# mode 0 folding
		>> fold(X_k_ij, mode=0, shape(K,I,J))
		>> X_i_j_k

		# mode 1 folding
		>> fold(X_i_jk, mode=1, shape(K,I,J))
		>> X_i_j_k

		# mode 2 folding
		>> fold(X_j_ki, mode=2, shape(K,I,J))
		>> X_i_j_k
	"""

	X = np.array(X)

	# checks
	if not isinstance(mode, int): raise ValueError(f'mode expected to be a number but got {type(mode).__name__()}')
	if mode not in range(len(shape)): raise ValueError(f'mode must be a valid tensor mode, but got mode={mode}. check manual for correct tensor modes')
	if not len(X.shape)==2: raise ValueError(f'X must be a matrix of size m*n, but got { X.shape }')
	if not len(shape)==3: raise ValueError(f'shape must be a three-way tensor shape, but got shape={shape}')
	if X.size!=np.prod(shape): raise ValueError(f'shape size {np.prod(shape)} is not consistent with tensor size {X.size}')

	# main
	if (mode==0): # (K,IJ) => (K,I,J)
		folded = np.array([x.reshape(shape[1],-1) for x in X])

	if (mode==1): # (I,JK) => (K,I,J)
		folded = np.array([X[:,shape[-1]*ind:shape[-1]*(ind+1)] for ind in range(X.shape[1]//shape[-1])])

	if (mode==2): # (J,KI) => (K,I,J)
		# folded = np.array([X[:,shape[1]*ind:shape[1]*(ind+1)].T for ind in range(X.shape[1]//shape[1])])
		folded = np.array([X_j_ki.T[k::K] for k in range(K)])

	return folded


class Tensor(np.ndarray):
    '''
    creates a tensor object from the given array matrix
    Args:
    -----
        input_array (ndarray): matrix of data values
        I (int): number of objects/units of the dataset
        J (int): number of variables in the dataset
        stacked (str, optional): placement of faces. possible values are 'row' or 'column'. Defaults to 'row'
    '''

    def __new__(cls, input_array, I, J, stacked='row'):
        obj = np.asarray(input_array)

        # checks
        if obj is None: raise ValueError('input_array cannot be None')
        if len(obj.shape)!=2: raise ValueError(f'input_array must be a matrix, but got shape {obj.shape}')

        # main
        if stacked=='column':
            obj = np.array([obj[:,J*ind:J*(ind+1)] for ind in range(obj.shape[1]//J)])  # reorder column stacked faces
        else:  # row
            obj = np.array([obj[I*ind:I*(ind+1),:] for ind in range(obj.shape[0]//I)])  # reorder rowstacked faces

        obj = np.asarray(obj).view(cls)
        return obj
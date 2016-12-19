import numpy as np

def sparse_tuple_from(sequences, n_classes, dtype=np.int32):
		

	idx = np.vstack([k for k, v in np.ndenumerate(sequences) if v > 0])
	val = np.asarray([v for k, v in np.ndenumerate(sequences) if v > 0])
	shape = np.asarray([sequences.shape[0], n_classes])
	print idx.shape
	return idx, val, shape

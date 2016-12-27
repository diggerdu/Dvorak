import numpy as np
import string

def sparse_tuple_from(sequences, n_classes, dtype=np.int32):
		

	idx = np.vstack([k for k, v in np.ndenumerate(sequences) if v > 0])
	val = np.asarray([v for k, v in np.ndenumerate(sequences) if v > 0])
	shape = np.asarray(sequences.shape)
	return idx, val, shape

def dense_from_sparse(sparse):
    idx, val, shape = sparse
    dense = np.zeros(shape)
    for i in range(idx.shape[0]):
        dense[tuple(idx[i])] = val[i]
    return dense

def decode_from_arr(arr):
    decode_dict = dict(enumerate(string.lowercase))
    assert len(arr.shape) == 2
    arr = arr.tolist()
    item_list = list()
    for item in arr:
        item = ''.join(map(lambda x:decode_dict[x], item))
        item_list.append(item)
    return item_list





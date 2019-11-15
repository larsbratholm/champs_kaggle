import pickle
import numpy as np
from typing import Iterable


def to_list(x):
    """
    If x is not a list, convert it to list
    """
    if isinstance(x, Iterable):
        return list(x)
    elif isinstance(x, np.ndarray):
        return x.tolist()
    else:
        return [x]


def expand_1st(x):
    """
    Adding an extra first dimension

    Args:
        x: (np.array)
    Returns:
         (np.array)
    """
    return np.expand_dims(x, axis=0)


def fast_label_binarize(value, labels):
    """Faster version of label binarize

    `label_binarize` from scikit-learn is slow when run 1 label at a time.
    `label_binarize` also is efficient for large numbers of classes, which is not
    common in `megnet`

    Args:
        value: Value to encode
        labels (list): Possible class values
    Returns:
        ([int]): List of integers
    """

    if len(labels) == 2:
        return [int(value == labels[0])]
    else:
        output = [0] * len(labels)
        if value in labels:
            output[labels.index(value)] = 1
        return output

def ring_to_vector(l):
    """
    Convert the ring sizes vector to a fixed length vector
    For example, l can be [3, 5, 5], meaning that the atom is involved
    in 1 3-sized ring and 2 5-sized ring. This function will convert it into
    [ 0, 0, 1, 0, 2, 0, 0, 0, 0, 0].

    Args:
        l: (list of integer) ring_sizes attributes

    Returns:
        (list of integer) fixed size list with the i-1 th element indicates number of
            i-sized ring this atom is involved in.
    """
    return_l = [0] * 9
    if l:
        for i in l:
            return_l[i - 1] += 1
    return return_l


def write_file_to_pickle(file, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(file, f, pickle.HIGHEST_PROTOCOL)

def read_pickle_from_file(pickle_file):
    with open(pickle_file,'rb') as f:
        x = pickle.load(f)
    return x

class Struct(object):
    def __init__(self, is_copy=False, **kwargs):
        self.add(is_copy, **kwargs)

    def add(self, is_copy=False, **kwargs):
        #self.__dict__.update(kwargs)

        if is_copy == False:
            for key, value in kwargs.items():
                setattr(self, key, value)
        else:
            for key, value in kwargs.items():
                try:
                    setattr(self, key, copy.deepcopy(value))
                    #setattr(self, key, value.copy())
                except Exception:
                    setattr(self, key, value)

    def __str__(self):
        return str(self.__dict__.keys())

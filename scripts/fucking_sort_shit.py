import numpy as np
import sys

def sort_shit(filename, sort_idx):
    data = np.loadtxt(filename, dtype=str, delimiter=',')
    for idx in sort_idx:
        if "." in data[1,idx]:
            type_ = float
        else:
            try:
                int(data[1,idx])
                type_ = int
            except:
                type_ = str
        subarray = data[1:,idx].astype(type_)
        order = np.argsort(subarray, kind='mergesort')
        data[1:] = data[order+1]

    np.savetxt(filename + ".sort", data, delimiter=',', fmt='%s')

if __name__ == "__main__":
    filename = sys.argv[1]
    sort_idx = np.asarray(sys.argv[2:], dtype=int)

    sort_shit(filename, sort_idx)


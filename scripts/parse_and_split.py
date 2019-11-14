import numpy as np
import re

"""
Creates train and test data files in the same format used on Kaggle.
"""

def split(filename, set_train, set_test):
    """
    Seperate input into header, train and test files.
    """
    train, test = [], []
    with open(filename) as f:
        header = f.readline().strip()
        for line in f:
            tokens = line.split(',')
            # Remove newline
            tokens[-1] = tokens[-1].strip()
            set_tokens = set(tokens)
            if set_tokens.intersection(set_train):
                train.append(tokens)
            elif set_tokens.intersection(set_test):
                test.append(tokens)
            else:
                # Molecules excluded by Kaggle
                pass
    return header, np.asarray(train), np.asarray(test)

def write_data(basename, train, test):
    np.savetxt(basename + "_train.csv", train, delimiter=',', fmt='%s')
    np.savetxt(basename + "_test.csv", test, delimiter=',', fmt='%s')

def add_index_and_header(header, train, test, idx):
    """
    Add id column and header
    """

    if idx:
        header = "id," + header

    train_index = np.empty((train.shape[0] + 1, train.shape[1] + int(idx)), dtype='<U32')
    test_index = np.empty((test.shape[0] + 1, test.shape[1] + int(idx)), dtype='<U32')

    train_index[0] = np.asarray(header.split(","), dtype='<U32')
    test_index[0] = train_index[0]

    if idx:
        train_index[1:,0] = np.arange(train.shape[0])
        test_index[1:,0] = np.arange(train.shape[0], train.shape[0] + test.shape[0])
        train_index[1:,1:] = train
        test_index[1:,1:] = test
    else:
        train_index[1:] = train
        test_index[1:] = test

    return train_index, test_index

def sort_data(data, sort_idx):
    """
    Sort the data array according to the given order of indices
    """
    for idx in sort_idx:
        if "." in data[0,idx]:
            type_ = float
        else:
            try:
                int(data[0,idx])
                type_ = int
            except:
                type_ = str
        subarray = data[:,idx].astype(type_)
        # Mergesort preserves order
        order = np.argsort(subarray, kind='mergesort')
        data[:] = data[order]

def process_data(set_train, set_test, basename, sort_idx, idx):
    """
    Splits a csv file into sorted and indexed train and test files.
    """

    header, train, test = split(basename + '.csv', set_train, set_test)
    sort_data(train, sort_idx)
    sort_data(test, sort_idx)
    train, test = add_index_and_header(header, train, test, idx)
    write_data(basename, train, test)

if __name__ == "__main__":
    with open('train_mols.txt') as f:
        train_mols = [line.strip() for line in f]
        train_mols_set = set(train_mols)
    with open('test_mols.txt') as f:
        test_mols = [line.strip() for line in f]
        test_mols_set = set(test_mols)

    #process_data(train_mols_set, test_mols_set, 'data', [2,1,0], idx=True)
    #process_data(train_mols_set, test_mols_set, 'scalar_coupling_contributions', [2,1,0], idx=False)
    #process_data(train_mols_set, test_mols_set, 'dipole_moments', [0], idx=False)
    #process_data(train_mols_set, test_mols_set, 'magnetic_shielding_tensors', [1,0], idx=False)
    #process_data(train_mols_set, test_mols_set, 'mulliken_charges', [1,0], idx=False)
    process_data(train_mols_set, test_mols_set, 'potential_energy', [0], idx=False)

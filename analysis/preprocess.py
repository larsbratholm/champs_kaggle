import numpy as np
import glob
import pickle
import gzip
import os

def parse_rank_and_name(filenames):
    rank = []
    name = []
    for filename in filenames:
        tokens = filename.split("/")[-2].split()
        rank.append(int(tokens[0]))
        if len(tokens) >= 1:
            name.append((" ".join(tokens[1:])))
        else:
            name.append("")
    return rank, name

def get_test_data(filename):
    id_to_type = {}
    id_to_idx = {}
    c = 0
    couplings = np.zeros(2505542)
    with gzip.open(filename, 'rt', encoding='utf-8') as f:
        next(f)
        for line in f:
            tokens = line.split(",")
            id_ = int(tokens[0])
            type_ = tokens[-2]
            coupling = float(tokens[-1])
            id_to_type[id_] = type_
            id_to_idx[id_] = c
            couplings[c] = coupling
            c += 1
    idx_to_id = {v: k for k, v in id_to_idx.items()}
    type_to_idx = {}
    for id_, type_ in id_to_type.items():
        if type_ not in type_to_idx:
            type_to_idx[type_] = []
        idx = id_to_idx[id_]
        type_to_idx[type_].append(idx)
    return id_to_type, id_to_idx, type_to_idx, couplings

def parse_submissions(filenames, id_to_idx):
    data = np.zeros((len(filenames), 2505542))
    for j, filename in enumerate(filenames):
        with open(filename) as f:
            # Catch all the weird quirks in formatting
            line = f.readline().strip().replace("\"", "").lower()
            header = np.asarray(line.split(","))
            id_column_idx = np.where(header == "id")[0][0]
            coupling_column_idx = 1 - id_column_idx
            for i, line in enumerate(f):
                tokens = line.strip().split(",")
                # Handles windows line breaks
                if len(tokens) < 2:
                    continue
                id_ = int(tokens[id_column_idx])
                idx = id_to_idx[id_]
                data[j,idx] = float(tokens[coupling_column_idx])
    return data

def get_score_by_type(data, couplings, type_to_idx):
    n = len(type_to_idx)

    scores = {}

    for type_, idx in type_to_idx.items():
        idx = np.asarray(idx)
        type_data = data[:,idx]
        type_couplings = couplings[idx]
        scores[type_] = np.log(np.mean(abs(type_data - type_couplings[None]), axis=1))
    return scores


if __name__ == "__main__":
    # Get script location
    script_dir = os.path.abspath(os.path.dirname(__file__))
    try:
        with open(script_dir + '/data/data.pkl', "rb") as f:
            scores, data, id_to_type, id_to_idx, type_to_idx, rank, name, filenames, couplings = pickle.load(f)
    except FileNotFoundError:
        filenames = glob.glob(script_dir + "/data/*/*.csv")
        if len(filenames) == 0:
            print("No submissions found in data folder")
            raise SystemExit
        filenames.sort()
        rank, name = parse_rank_and_name(filenames)
        id_to_type, id_to_idx, type_to_idx, couplings = get_test_data(script_dir + "/data/data_test.csv.gz")
        data = parse_submissions(filenames, id_to_idx)
        scores = get_score_by_type(data, couplings, type_to_idx)
        with open(script_dir + '/data/data.pkl', "wb") as f:
            pickle.dump((scores, data, id_to_type, id_to_idx, type_to_idx, rank, name, filenames, couplings),f,-1)
